
class RagIndexError(Exception):
    pass


class SearchError(Exception):
    pass


class InputSingleQueryError(SearchError):
    def __init__(self) -> None:
        super().__init__("uv run -m src search <query> "
                         "[--k N] [--save_directory PATH]")


class AnswerError(Exception):
    pass


class EvaluateError(Exception):
    pass
