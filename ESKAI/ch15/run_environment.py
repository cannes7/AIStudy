import argparse
import gym

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Run an environment')
    parser.add_argument('--input-env', dest='input_env', required=True,
                        choices=['cartpole', 'mountaincar', 'pendulum', 'taxi', 'lake'],
                        help='Specify the name of the environment')
    return parser

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_env = args.input_env

    # 환경 이름 맵핑 (최신 버전 사용)
    name_map = {'cartpole': 'CartPole-v1',
                'mountaincar': 'MountainCar-v0',
                'pendulum': 'Pendulum-v0',
                'taxi': 'Taxi-v3',
                'lake': 'FrozenLake-v0'}

    # 환경 생성 (렌더 모드 설정)
    env = gym.make(name_map[input_env], render_mode='human')
    observation = env.reset()

    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # 무작위 행동 선택
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation = env.reset()

    env.close()  # 환경 종료
