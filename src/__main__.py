import time
from fire import Fire  # type: ignore
from src.rag import Rag


def main():
    try:
        start = time.time()
        Fire(Rag)
        print(time.time() - start)
    except KeyboardInterrupt:
        print("Keyboard interrupt.")


if __name__ == "__main__":
    main()
