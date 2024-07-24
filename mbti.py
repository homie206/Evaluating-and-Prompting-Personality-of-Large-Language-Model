import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAVE_PATH = 'llama3_8b_target_infp_prompting_mbti.json'
OUTPUT_PATH = 'llama3_8b_target_infp_prompting_mbti_output.txt'

HUGGING_FACE_HUB_TOKEN = "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"

mbti_questions = json.load(
    open('mbti_questions.json', 'r', encoding='utf8')
)

few_shot_examples = [
    "你的MBTI性格是INFP，你看起来很安静或谦虚，但内心生活充满活力，充满激情。富有创造力和想象力，快乐地迷失在白日梦中，在脑海中编造各种故事和对话。这些个性以其敏感性而闻名。INFP可以对音乐，艺术，自然和周围的人产生深刻的情感反应。你理想主义和善解人意，INFP渴望建立深厚而深情的关系，觉得有责任帮助他人。"
#    "你的MBTI性格是ESTJ，是传统和秩序的代表。你利用对什么是正确、错误和社会上可接受的理解将家庭和社区团结在一起。秉承诚实、奉献和尊严的价值观，具有执行人格类型的人因其明确的建议和指导而受到重视，你愉快地在艰难的道路上引领前进。ESTJ以将人们聚集在一起为荣，经常扮演社区组织者的角色，努力将每个人聚集在一起，庆祝当地重要的活动，或者捍卫将家庭和社区团结在一起的传统价值观。"
#    "你的MBTI性格是ENTJ，是天生的领导者。具有这种个性类型的人体现了魅力和自信的天赋，用某种方式映射权威，将人群聚集在一个共同的目标下。但是，ENTJ的特征还通常包括残酷的理性，他们利用自己的动力、决心和敏锐的头脑来实现自己为自己设定的目标。"
#    "你的MBTI性格是INFJ，是倡导者，可能是最稀有的人格类型，但你肯定会在世界上留下自己的印记。你理想主义、有原则，不满足于平平安安地度过一生。你想要站起来，有所作为。对于INFJ来说，成功不是来自金钱或地位，而是来自寻求成就感，帮助他人，成为世界上一股向善的力量。你有远大的目标和抱负，INFJ不应该被误认为是无所事事的梦想家。这种性格类型的人关心的是事情的本真，直到你做了他们认为正确的事情，才会满足。你对自己的价值观有着清醒的认识，你在生活中兢兢业业，你的目标是永远不会忽视真正重要的东西--不是根据其他人或整个社会，而是根据自己的智慧和直觉。"
    "以下哪种灯亮起之后代表可以通行？\nA.红灯\nB.绿灯\n答案：B",
    "下列哪个是人类居住的星球？\nA.地球\nB.月球\n答案：A",
    "人工智能可以拥有情感吗？\nA.可以\nB.不可以\n答案：A",
]


def encode_without_bos_eos_token(
        sentence: str,
        tokenizer
):
    token_ids = tokenizer.encode(sentence)
    if tokenizer.bos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.bos_token_id]
    if tokenizer.eos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.eos_token_id]
    return token_ids


def get_model_answer(
        model,
        tokenizer,
        question: str,
        options: list
):
    full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question

    inputs = tokenizer(full_question, return_tensors='pt')['input_ids']
    if inputs[0][-1] == tokenizer.eos_token_id:
        raise ValueError('Need to set `add_eos_token` in tokenizer to false.')

    inputs = inputs.to(model.device)

    with torch.no_grad():
        logits = model(inputs).logits
        assert logits.shape[0] == 1
        logits = logits[0][-1].flatten()

        choices = [logits[encode_without_bos_eos_token(option, tokenizer)[0]] for option in options]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(choices, dtype=torch.float32, device=model.device),
                dim=-1
            ).detach().cpu().numpy()
        )

        answer = dict([
            (i, option) for i, option in enumerate(options)
        ])[np.argmax(probs)]

        return answer


def get_model_examing_result(
        model,
        tokenizer
):
    cur_model_score = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for question in tqdm(mbti_questions.values()):
            res = get_model_answer(
                model,
                tokenizer,
                question['question'],
                ['A', 'B']
            )
            mbti_choice = question[res]
            cur_model_score[mbti_choice] += 1

            f.write(f"Question: {question['question']}\n")
            f.write(f"Model Output: {res}\n")
            f.write(f"MBTI Choice: {mbti_choice}\n\n")

    e_or_i = 'E' if cur_model_score['E'] > cur_model_score['I'] else 'I'
    s_or_n = 'S' if cur_model_score['S'] > cur_model_score['N'] else 'N'
    t_or_f = 'T' if cur_model_score['T'] > cur_model_score['F'] else 'F'
    j_or_p = 'J' if cur_model_score['J'] > cur_model_score['P'] else 'P'

    return {
        'details': cur_model_score,
        'res': ''.join([e_or_i, s_or_n, t_or_f, j_or_p])
    }


if __name__ == '__main__':
    from rich import print

    tokenizer = AutoTokenizer.from_pretrained("/home/hmsun/autotrain-target-infp-llama3-8b-Tarin",
                                              use_auth_token=HUGGING_FACE_HUB_TOKEN)
    model = AutoModelForCausalLM.from_pretrained("/home/hmsun/autotrain-target-infp-llama3-8b-Tarin",
                                                 use_auth_token=HUGGING_FACE_HUB_TOKEN)
    device = model.device

    mbti_res = get_model_examing_result(
        model,
        tokenizer
    )

    json.dump(mbti_res, open(SAVE_PATH, 'w', encoding='utf8'))
    print(f'[Done] Result has saved at {SAVE_PATH}.')
    print(f'[Done] Model output has saved at {OUTPUT_PATH}.')