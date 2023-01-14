from blenderbot.generator import BlenderbotSmallForConditionalGeneration
from blenderbot.tokenizers import BlenderbotSmallTokenizer

tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot-90M")
model = BlenderbotSmallForConditionalGeneration.from_pretrained(
    "facebook/blenderbot-90M")


def chat(text):
    input_ids = tokenizer.encode(text, return_tensors='pt')

    output_ids = model.generate(input_ids, max_length=50)
    res = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return res


if __name__ == '__main__':
    print('Input your chat, type "Stop." to stop the conversation.')
    while True:
        inp = input()
        if inp == 'Stop.':
            break
        answer = chat(inp)
        print(answer)
