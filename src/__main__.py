from fire import Fire  # type: ignore
from src.rag import Rag


def main():
    Fire(Rag)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Keyboard interrupt.")
    except Exception as e:
        # raise e
        print(f"\033[1;31m[{type(e).__name__}]\033[0m {e}")
