from fire import Fire  # type: ignore
from src.rag import Rag


def main():
    try:
        Fire(Rag)
    except KeyboardInterrupt:
        print("Keyboard interrupt.")


if __name__ == "__main__":
    main()
