from __future__ import annotations

import subprocess

from uci_parser import (
    SearchResult,
    StaticEvalResult,
    parse_display_fen,
    parse_search_output,
    parse_static_eval,
)


class StockfishSession:
    def __init__(
        self,
        executable: str,
        eval_file: str,
        *,
        threads: int = 1,
        hash_mb: int = 32,
        multipv: int = 1,
        show_wdl: bool = True,
    ):
        self.executable = executable
        self.eval_file = eval_file
        self.current_multipv = multipv

        self.proc = subprocess.Popen(
            [executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Failed to open Stockfish pipes")

        self._send("uci")
        self._read_until_exact("uciok")

        self.set_option("Threads", threads)
        self.set_option("Hash", hash_mb)
        self.set_option(
            "UCI_ShowWDL",
            "true" if show_wdl else "false",
        )
        self.set_option("MultiPV", multipv)
        self.set_option("EvalFile", eval_file)

        self.ready()

    # --------------------------------------------------------
    # Low-level UCI transport
    # --------------------------------------------------------

    def _send(self, command: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"Stockfish already terminated with "
                f"code {self.proc.returncode}"
            )

        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def _readline(self) -> str:
        line = self.proc.stdout.readline()

        if line == "":
            raise RuntimeError(
                "Unexpected EOF from Stockfish"
            )

        return line.rstrip("\r\n")

    def _read_until_exact(self, target: str) -> list[str]:
        lines = []

        while True:
            line = self._readline()
            lines.append(line)

            if line == target:
                return lines

    # --------------------------------------------------------
    # UCI setup
    # --------------------------------------------------------

    def set_option(self, name: str, value) -> None:
        self._send(
            f"setoption name {name} value {value}"
        )

    def ready(self) -> None:
        self._send("isready")
        self._read_until_exact("readyok")

    def new_game(self, *, clear_hash: bool = True) -> None:
        self._send("ucinewgame")

        if clear_hash:
            self._send("setoption name Clear Hash")

        self.ready()

    def set_multipv(self, multipv: int) -> None:
        if multipv < 1:
            raise ValueError("MultiPV must be >= 1")

        if multipv != self.current_multipv:
            self.set_option("MultiPV", multipv)
            self.current_multipv = multipv
            self.ready()

    # --------------------------------------------------------
    # Position / eval
    # --------------------------------------------------------

    def set_position(self, fen: str) -> None:
        self._send(f"position fen {fen}")

    def fen_after_moves(
        self,
        fen: str,
        moves: list[str],
    ) -> str:
        if not moves:
            return fen

        command = (
            "position fen "
            + fen
            + " moves "
            + " ".join(moves)
        )

        self._send(command)
        self._send("d")

        lines = []

        while True:
            line = self._readline()
            lines.append(line)

            if line.startswith("Checkers:"):
                break

        child_fen = parse_display_fen(lines)

        if child_fen is None:
            raise RuntimeError(
                "Could not parse FEN from Stockfish d output"
            )

        return child_fen

    def static_eval(
        self,
        fen: str,
    ) -> tuple[StaticEvalResult, list[str]]:
        self.set_position(fen)
        self._send("eval")

        lines = []

        while True:
            line = self._readline()
            lines.append(line)

            if line.startswith("Final evaluation"):
                break

        return parse_static_eval(lines), lines

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    def search(
        self,
        fen: str,
        *,
        nodes: int,
        multipv: int = 1,
        searchmoves: list[str] | None = None,
    ) -> tuple[SearchResult, list[str]]:
        if nodes <= 0:
            raise ValueError("nodes must be > 0")

        self.set_multipv(multipv)
        self.set_position(fen)

        command = f"go nodes {nodes}"

        # Stockfish requires searchmoves to be last.
        if searchmoves:
            command += " searchmoves " + " ".join(searchmoves)

        self._send(command)

        lines = []

        while True:
            line = self._readline()
            lines.append(line)

            if line.startswith("bestmove "):
                break

        return parse_search_output(
            lines,
            expected_multipv=multipv,
        ), lines

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                try:
                    self._send("quit")
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        finally:
            # Explicitly close subprocess pipe wrappers.
            # Otherwise Python may emit ResourceWarning even
            # after the Stockfish process itself has exited.
            for stream in (
                self.proc.stdin,
                self.proc.stdout,
            ):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
