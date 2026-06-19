class SearchError(Exception):
    pass


class InputSingleQueryError(SearchError):
    def __init__(self):
        super().__init__("uv run -m src search <query> "
                         "[--k N] [--save_directory PATH]")


# class InputSearchDatasetError(SearchError):
#     def __init__(self) -> None:
#         super().__init__("uv run -m src search_dataset [--dataset_path PATH]"
#                          "[--k N] [--save_directory PATH]")

class RagIndexError(Exception):
    pass
