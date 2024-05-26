import argparse
import gym

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Run an environment')
    parser.add_argument('--input-env', dest='input_env', required=True,
                        choices=['cartpole', 'mountaincar', 'pendulum'],
                        help='Specify the name of the environment')
    return parser

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_env = args.input_env

    name_map = {'cartpole': 'CartPole-v1',
                'mountaincar': 'MountainCar-v0',
                'pendulum': 'Pendulum-v0'}

    # 환경 생성 (렌더 모드 설정)
    env = gym.make(name_map[input_env], render_mode='human')

    # 환경 초기화
    for _ in range(20):
        observation = env.reset()

        # 100번 반복
        for i in range(100):
            # 환경 렌더링
            env.render()

            # 현재 observation 출력
            print(observation)

            # 무작위 행동 선택
            action = env.action_space.sample()

            # 행동에 따른 observation, 보상, 종료 상태, 추가 정보 반환
            observation, reward, terminated, truncated, info = env.step(action)

            # 에피소드 종료 체크
            if terminated or truncated:
                print(f'Episode finished after {i+1} timesteps')
                break

    # 환경 종료
    env.close()
