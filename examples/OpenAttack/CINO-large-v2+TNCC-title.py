import sys
import os

sys.path.append(os.getcwd())

import OpenAttack
import open_attack
from datasets import load_dataset


def dataset_mapping(data):
    return {
        "x": data["text"],
        "y": data["label"]
    }


def main():
    print("Loading Attacker...")
    # attacker = open_attack.attackers.TSAttacker(lang="tibetan")
    # attacker = open_attack.attackers.TSTricker_s(lang="tibetan")
    # attacker = open_attack.attackers.TSTricker_w(lang="tibetan")
    # attacker = open_attack.attackers.TSCheater_s(lang="tibetan")
    # attacker = open_attack.attackers.TSCheater_w(lang="tibetan")
    attacker = open_attack.attackers.TSCheaterPlus(lang="tibetan")

    print("Loading Victim ...")
    victim = OpenAttack.loadVictim("XLM-RoBERTa.CINO-large-v2+TNCC-title")

    print("Loading Dataset ...")
    dataset = load_dataset(path=os.path.join(os.getcwd(), "data", "Dataset.Loader", "TNCC-title.py"), split="test",
                           trust_remote_code=True).map(function=dataset_mapping)

    print("Start Attack!")
    attack_eval = open_attack.AttackEval(attacker, victim, "tibetan", metrics=[])
    attack_eval.eval(dataset, visualize=True, progress_bar=True)


if __name__ == "__main__":
    main()
