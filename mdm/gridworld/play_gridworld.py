import time

from pynput.keyboard import KeyCode, Listener, Key

from mdm.gridworld.gridworld import Gridworld
from mdm.utils.utils import here


def main():
    env = Gridworld.from_cleartext(here() / '8x8_v0.mapdata')

    current_a = None
    actions = [KeyCode(char=x) for x in ('w', 'd', 's', 'a')]

    def on_press(key):
        nonlocal current_a
        if key in actions:
            current_a = actions.index(key)
        else:
            current_a = None

    def on_release(key):
        nonlocal current_a
        current_a = None

    with Listener(on_press=on_press, on_release=on_release) as listener:
        env.reset()
        while True:
            time.sleep(0.2)
            env.render()
            if current_a is not None:
                o, r, term, trunc, _ = env.step(current_a)
                env.render()

                print(f'r: {r} | term: {term} | trunc: {trunc}')

                if term or trunc:
                    env.reset()
        listener.join()


if __name__ == '__main__':
    main()
