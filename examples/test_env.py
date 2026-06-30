from gymtorax import TestEnv

if __name__ == "__main__":
    env = TestEnv()
    env.reset()

    for i in range(10):
        env.step({"Ip": [3e6]})
