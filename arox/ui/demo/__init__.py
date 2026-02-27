import argparse
import os

import yaml

from arox.ui import TUIByIO


class DemoTUI(TUIByIO):
    def __init__(self, app_name, skip_read=True):
        super().__init__(app_name)
        self.skip_read = skip_read

    def on_mount(self) -> None:
        io_generator = IOGenerator(self.io_channel, skip_read=self.skip_read)
        self.run_worker(io_generator.run, name="io generator", exclusive=True)


class IOGenerator:
    def __init__(self, io_channel, skip_read=True):
        self.io_channel = io_channel
        self.skip_read = skip_read

    async def stream_content(self, content, interval):
        import asyncio

        for i in range(0, len(content), 2):
            yield content[i : i + 2]
            await asyncio.sleep(interval)

    async def do_action(self, io_channel, io_actions):
        for io_action in io_actions:
            if io_action["action"] == "read":
                if not self.skip_read:
                    async for _ in io_channel.read():
                        break
            elif io_action["action"] == "write":
                content = io_action.get("content", "")
                interval = io_action.get("stream_interval", 0)
                interval = float(interval)
                if interval == 0.0:
                    await io_channel.agent_send(content)
                else:
                    await io_channel.agent_send(self.stream_content(content, interval))
            elif io_action["action"] == "create_sub_channel":
                sub_channel = io_channel.create_sub_channel(
                    io_action["type"], io_action.get("title")
                )
                await self.do_action(sub_channel, io_action["actions"])

    async def run(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "io_example.yaml")

        with open(file_path, "r", encoding="utf-8") as file:
            io_actions = yaml.safe_load(file)
            await self.do_action(self.io_channel, io_actions)


def main():
    parser = argparse.ArgumentParser(description="UI demo")
    parser.add_argument(
        "--skip-read",
        action="store_true",
        default=False,
        help="Skip read actions",
    )

    args = parser.parse_args()

    app = DemoTUI("UI demo", skip_read=args.skip_read)
    app.run()


if __name__ == "__main__":
    main()
