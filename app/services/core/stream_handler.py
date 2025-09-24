# handler.py
import re
import time
from typing import List, AsyncIterator, AsyncGenerator, Union

from langchain_core.messages import BaseMessageChunk
from sse_starlette.sse import ServerSentEvent

from app.utils.helpers import ping, send, event

# 문장 경계(한/영 구두점 포함)
_SENT_END = re.compile(r"(?:[\.!\?。？！…]+(?:\s|$))")
_KO_TAILS = ("다.", "요.", "죠.", "네.", "습니다.", "됩니다.", "합니다.", "예요.", "이에요.")


class CoalescingHandler:
    """
    LangChain `astream()`이 내주는 `BaseMessageChunk` 스트림을 받아서:
      - 첫 토큰은 즉시 전송(반응성 확보)
      - 이후 토큰은 문장 경계 / 최소 글자수 / 최대 지연 기준으로 코얼레싱 전송
      - 아이들 시간에는 주기적으로 heartbeat(comment) 전송
    ※ meta/end 트레일러는 바깥 파이프라인에서 공통 처리 권장.
    """

    def __init__(
            self,
            stream: AsyncIterator[BaseMessageChunk],
            *,
            min_chars: int = 150,
            max_latency_ms: int = 200,
            heartbeat_secs: int = 15,
    ):
        self.stream = stream
        self.min_chars = int(min_chars)
        self.max_latency_ms = int(max_latency_ms)
        self.heartbeat_secs = int(heartbeat_secs)

        self._buffer: str = ""  # 아직 내보내지 않은 누적 텍스트
        self._parts: List[str] = []  # 전체 답변 조각(캐시/로그용)
        self._first_sent: bool = False

        now = self._now_s()
        self._last_flush_ms = now * 1000.0  # 마지막 flush 시각(ms)
        self._last_hb_at = now  # 마지막 heartbeat 시각(s)

    # 외부에서: `async for frame in CoalescingHandler(...): yield frame`
    def __aiter__(self) -> AsyncIterator[Union[str, ServerSentEvent]]:
        return self._generate()

    async def _generate(self) -> AsyncGenerator[Union[str, ServerSentEvent], None]:
        try:
            async for chunk in self.stream:
                content = getattr(chunk, "content", "") or ""

                # 토큰이 없으면 heartbeat만 유지
                if not content:
                    if self._should_send_heartbeat():
                        yield ping()
                        self._last_hb_at = self._now_s()
                    continue

                # 첫 토큰은 즉시 전송(반응성)
                if not self._first_sent:
                    yield send({"answer": content})
                    self._first_sent = True
                    self._last_flush_ms = self._now_ms()
                    self._last_hb_at = self._now_s()
                    self._parts.append(content)
                    continue

                # 이후 토큰은 버퍼에 쌓아 코얼레싱
                self._buffer += content
                self._parts.append(content)

                now_ms = self._now_ms()
                if self._should_flush_sentence(now_ms):
                    yield send({"answer": self._buffer})
                    self._buffer = ""
                    self._last_flush_ms = now_ms
                    self._last_hb_at = self._now_s()

        except Exception as e:
            # 클라이언트 디버깅용(필요 없으면 제거 가능)
            yield event("error", {"message": str(e)})
            raise
        finally:
            # 남은 버퍼가 있으면 마지막으로 전송
            if self._buffer:
                yield send({"answer": self._buffer})

    # 전체 답변 문자열(캐시/로그용)
    def get_answer(self) -> str:
        return "".join(self._parts)

    # ===== 내부 헬퍼 =====
    def _should_flush_sentence(self, now_ms: float) -> bool:
        # 1) 최소 글자수 도달
        if len(self._buffer) >= self.min_chars:
            return True
        # 2) 문장 경계 + 하한 길이(짧은 문장 남발 방지)
        if len(self._buffer) >= max(24, self.min_chars // 4) and self._has_sentence_boundary(self._buffer):
            return True
        # 3) 최대 지연 도달
        if (now_ms - self._last_flush_ms) >= self.max_latency_ms and self._buffer:
            return True
        return False

    def _should_send_heartbeat(self) -> bool:
        return (self._now_s() - self._last_hb_at) >= self.heartbeat_secs

    @staticmethod
    def _has_sentence_boundary(text: str) -> bool:
        if not text:
            return False
        if _SENT_END.search(text):
            return True
        return any(text.endswith(t) for t in _KO_TAILS)

    @staticmethod
    def _now_s() -> float:
        return time.monotonic()

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0
