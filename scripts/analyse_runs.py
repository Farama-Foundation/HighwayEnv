from rl_agents.trainer.analyzer import RunAnalyzer

if __name__ == '__main__':
    RunAnalyzer(['out/HighwayEnv/DQNPytorchAgent/e50000',
                 'out/HighwayEnv/MCTSAgent/avg_i25_e80',
                 'out/HighwayEnv/MCTSAgent/avg_i75_e80',
                 'out/HighwayEnv/MCTSWithPriorPolicyAgent/eps09_i25_t10',
                 'out/HighwayEnv/MCTSWithPriorPolicyAgent/eps08_i25_t10',
                 'out/HighwayEnv/MCTSWithPriorPolicyAgent/eps03_i25_t10'])


