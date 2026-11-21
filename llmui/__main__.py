import argparse

from .app import build


def main():
    ap = argparse.ArgumentParser(description="Gradio UI for KoboldCpp")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build().queue().launch(server_name=a.host, server_port=a.port)


if __name__ == "__main__":
    main()
