//! The exec operations seam: where a shell command actually executes.
//!
//! The bash tool keeps all tool logic — schema, destructive-command heuristic,
//! truncation, timeout semantics — and performs the raw process execution
//! through [`ExecOps`], carried by its execution context. The default,
//! [`LocalExecOps`], is the host spawn path; a sandbox backend implements the
//! same trait without touching the tool.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

/// A sink an [`ExecOps`] implementation streams output chunks into as the
/// command runs (the bash tool forwards each chunk through `ctx.update`).
/// Chunks must land on UTF-8 boundaries.
pub type ExecOutput = tokio::sync::mpsc::Sender<String>;

/// What one command run produced: the full output and how the process ended.
#[derive(Debug)]
pub struct ExecOutcome {
    /// The complete output — stdout then stderr, lossily decoded.
    pub output: String,
    /// The exit code; `None` when the process was killed instead of exiting.
    pub exit_code: Option<i32>,
    /// Whether the implementation killed the process because `timeout` elapsed.
    pub timed_out: bool,
}

/// Runs a command in a working directory — the narrow seam between the bash
/// tool and the machinery that actually spawns processes (the local host
/// today; a container or microVM later). Per-session: carried by the tool
/// context and configured through the runtime builder.
#[async_trait]
pub trait ExecOps: Send + Sync {
    /// Run `command` through a shell in `cwd`, streaming output chunks into
    /// `output` as they arrive. The implementation enforces `timeout` —
    /// killing the process when it elapses — and must also kill the process
    /// when the returned future is dropped (cancellation). `Err` is reserved
    /// for failing to run the command at all.
    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        timeout: Option<Duration>,
        output: ExecOutput,
    ) -> Result<ExecOutcome>;
}

/// The default [`ExecOps`]: spawn `sh -c` on the host, in `cwd`. This is the
/// bash tool's original spawn path, unchanged in behavior.
pub struct LocalExecOps;

#[async_trait]
impl ExecOps for LocalExecOps {
    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        timeout: Option<Duration>,
        output: ExecOutput,
    ) -> Result<ExecOutcome> {
        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c")
            .arg(command)
            .current_dir(cwd)
            .kill_on_drop(true)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Pump stdout and stderr concurrently: stream each line through the
        // sink as it arrives and accumulate the raw bytes, so the final output
        // is byte-for-byte what `output()` would have produced (stdout then
        // stderr, lossily decoded). On a timeout — or when the caller drops
        // this future — the in-flight work drops and `kill_on_drop` kills sh.
        let work = async {
            let mut child = cmd
                .spawn()
                .map_err(|e| anyhow!("failed to run command: {e}"))?;
            let stdout = child.stdout.take().expect("piped stdout");
            let stderr = child.stderr.take().expect("piped stderr");
            let (out_bytes, err_bytes) =
                tokio::join!(pump(stdout, &output), pump(stderr, &output));
            let status = child
                .wait()
                .await
                .map_err(|e| anyhow!("failed to run command: {e}"))?;
            Ok::<_, anyhow::Error>((out_bytes, err_bytes, status))
        };
        let (out_bytes, err_bytes, status) = match timeout {
            Some(t) => match tokio::time::timeout(t, work).await {
                Ok(done) => done?,
                Err(_) => {
                    return Ok(ExecOutcome {
                        output: String::new(),
                        exit_code: None,
                        timed_out: true,
                    });
                }
            },
            None => work.await?,
        };

        let mut text = String::from_utf8_lossy(&out_bytes).into_owned();
        text.push_str(&String::from_utf8_lossy(&err_bytes));
        Ok(ExecOutcome {
            output: text,
            exit_code: status.code(),
            timed_out: false,
        })
    }
}

/// Drain a child pipe to EOF, accumulating the raw bytes and streaming each
/// line (lossily decoded) into the sink as it arrives. Reading by line keeps
/// chunks on UTF-8 boundaries (`\n` never splits a multibyte char) and lets
/// the caller show output live without changing the final, authoritative
/// output.
async fn pump<R: AsyncRead + Unpin>(reader: R, output: &ExecOutput) -> Vec<u8> {
    let mut buf = BufReader::new(reader);
    let mut acc: Vec<u8> = Vec::new();
    loop {
        let start = acc.len();
        match buf.read_until(b'\n', &mut acc).await {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let _ = output
                    .send(String::from_utf8_lossy(&acc[start..]).into_owned())
                    .await;
            }
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether `pid` is still alive and not yet reaped (signal 0 probes
    /// without killing).
    fn alive(pid: &str) -> bool {
        std::process::Command::new("kill")
            .args(["-0", pid])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Run the command through [`LocalExecOps`]; it writes its shell's pid to
    /// a file in a fresh temp dir before doing anything else. Returns the dir
    /// and the pid once it has landed.
    async fn spawn_with_pidfile(
        tag: &str,
        rest: &str,
    ) -> (std::path::PathBuf, String) {
        let dir = std::env::temp_dir()
            .join(format!("tapir-exec-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        (dir, format!("echo $$ > pid; {rest}"))
    }

    /// Poll until `pid` is gone, failing after a bounded wait (the kill is
    /// asynchronous: SIGKILL on drop, reaped by tokio in the background).
    async fn assert_killed(pid: &str) {
        for _ in 0..100 {
            if !alive(pid) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("process {pid} is still alive — the seam did not kill it");
    }

    #[tokio::test]
    async fn local_exec_timeout_kills_the_underlying_process() {
        let (dir, command) = spawn_with_pidfile("timeout", "sleep 30").await;
        let (tx, _rx) = tokio::sync::mpsc::channel(64);

        let outcome = LocalExecOps
            .exec(&command, &dir, Some(Duration::from_millis(200)), tx)
            .await
            .expect("a timed-out run is an outcome, not a failure to run");
        assert!(outcome.timed_out, "the ops report the timeout");
        assert_eq!(
            outcome.exit_code, None,
            "a killed process has no exit code"
        );

        let pid = std::fs::read_to_string(dir.join("pid"))
            .unwrap()
            .trim()
            .to_string();
        assert_killed(&pid).await;
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn dropping_a_local_exec_kills_the_underlying_process() {
        // Cancellation is the future being dropped — exactly what happens when
        // the engine aborts a turn's task mid-call. The contract: the process
        // must not outlive the exec.
        let (dir, command) = spawn_with_pidfile("cancel", "sleep 30").await;
        let (tx, _rx) = tokio::sync::mpsc::channel(64);

        let exec = tokio::spawn({
            let dir = dir.clone();
            async move { LocalExecOps.exec(&command, &dir, None, tx).await }
        });
        // Wait for the command to be running (its pid has landed), then cancel.
        let pid = loop {
            if let Ok(pid) = std::fs::read_to_string(dir.join("pid"))
                && !pid.trim().is_empty()
            {
                break pid.trim().to_string();
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        exec.abort();

        assert_killed(&pid).await;
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use super::*;

    /// One exec call as the fake saw it.
    #[derive(Clone, Debug)]
    pub struct SeenExec {
        pub command: String,
        pub cwd: PathBuf,
        pub timeout: Option<Duration>,
    }

    /// A scripted [`ExecOps`]: records every call and replays fixed output —
    /// streamed through the sink chunk by chunk, like a real backend, then
    /// returned whole in the outcome.
    pub struct FakeExecOps {
        calls: Mutex<Vec<SeenExec>>,
        chunks: Vec<String>,
        exit_code: Option<i32>,
    }

    impl FakeExecOps {
        /// A fake whose every exec streams `output` as one chunk and exits
        /// with `exit_code`.
        pub fn ok(output: &str, exit_code: i32) -> Self {
            Self::streaming(&[output], exit_code)
        }

        /// A fake whose every exec streams these chunks in order.
        pub fn streaming(chunks: &[&str], exit_code: i32) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                chunks: chunks.iter().map(|c| c.to_string()).collect(),
                exit_code: Some(exit_code),
            }
        }

        /// Every exec call recorded so far.
        pub fn calls(&self) -> Vec<SeenExec> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ExecOps for FakeExecOps {
        async fn exec(
            &self,
            command: &str,
            cwd: &Path,
            timeout: Option<Duration>,
            output: ExecOutput,
        ) -> Result<ExecOutcome> {
            self.calls.lock().unwrap().push(SeenExec {
                command: command.to_string(),
                cwd: cwd.to_path_buf(),
                timeout,
            });
            for chunk in &self.chunks {
                let _ = output.send(chunk.clone()).await;
            }
            Ok(ExecOutcome {
                output: self.chunks.concat(),
                exit_code: self.exit_code,
                timed_out: false,
            })
        }
    }
}
