
import json
import os

# N=30 Matched Triples (Ja, Romaji, En)
DATA = [
    ("猫は犬を追いかける。", "Neko wa inu o oikakeru.", "The cat chases the dog."),
    ("私はリンゴを食べる。", "Watashi wa ringo o taberu.", "I eat an apple."),
    ("彼女は学校へ行く。", "Kanojo wa gakkou e iku.", "She goes to school."),
    ("鳥は空を飛ぶ。", "Tori wa sora o tobu.", "Birds fly in the sky."),
    ("プログラミングは楽しい。", "Puroguramingu wa tanoshii.", "Programming is fun."),
    ("今日の天気はいいです。", "Kyou no tenki wa ii desu.", "The weather is good today."),
    ("新しいパソコンを買いたい。", "Atarashii pasokon o kaitai.", "I want to buy a new computer."),
    ("数学は宇宙の言語だ。", "Suugaku wa uchuu no gengo da.", "Mathematics is the language of the universe."),
    ("人工知能は世界を変えている。", "Jinkou chinou wa sekai o kaete iru.", "Artificial intelligence is changing the world."),
    ("東京は日本の首都だ。", "Toukyou wa Nihon no shuto da.", "Tokyo is the capital of Japan."),
    ("ラーメンは美味しい。", "Raamen wa oishii.", "Ramen is delicious."),
    ("機械学習を勉強する。", "Kikai gakushuu o benkyou suru.", "I study machine learning."),
    ("ニューラルネットワークは強力だ。", "Nyuuraru nettowaaku wa kyouryoku da.", "Neural networks are powerful."),
    ("早い茶色の狐が怠け者の犬を飛び越える。", "Hayai chairo no kitsune ga namakemono no inu o tobikoeru.", "The quick brown fox jumps over the lazy dog."),
    ("沈黙は金。", "Chinmoku wa kin.", "Silence is golden."),
    ("光陰矢の如し。", "Kouin ya no gotoshi.", "Time flies like an arrow."),
    ("知識は力なり。", "Chishiki wa chikara nari.", "Knowledge is power."),
    ("健康は富。", "Kenkou wa tomi.", "Health is wealth."),
    ("継続は力なり。", "Keizoku wa chikara nari.", "Perseverance is power."),
    ("ローマは一日にして成らず。", "Rooma wa ichinichi ni shite narazu.", "Rome was not built in a day."),
    ("彼はギターを弾く。", "Kare wa gitaa o hiku.", "He plays the guitar."),
    ("私たちは平和を愛する。", "Watashitachi wa heiwa o aisuru.", "We love peace."),
    ("科学は進歩する。", "Kagaku wa shinpo suru.", "Science advances."),
    ("歴史は繰り返す。", "Rekishi wa kurikaesu.", "History repeats itself."),
    ("夢を追いかける。", "Yume o oikakeru.", "Chasing dreams."),
    ("失敗は成功の母。", "Shippai wa seikou no haha.", "Failure is the mother of success."),
    ("読書は心を豊かにする。", "Dokusho wa kokoro o yutaka ni suru.", "Reading enriches the mind."),
    ("音楽は世界共通語だ。", "Ongaku wa sekai kyoutsuugo da.", "Music is a universal language."),
    ("自然を守ろう。", "Shizen o mamorou.", "Let's protect nature."),
    ("未来は明るい。", "Mirai wa akarui.", "The future is bright."),
]

def create_full_controls():
    os.makedirs("data/topology_control_full", exist_ok=True)
    
    ja_data = []
    romaji_data = []
    en_data = []
    
    for i, (ja, rom, en) in enumerate(DATA):
        # Japanese (Kana)
        ja_data.append({
            "text": ja,
            "lang": "ja", 
            "type": "active",
            "id": f"pair_{i}_ja"
        })
        # Romaji
        romaji_data.append({
            "text": rom,
            "lang": "ja_romaji",
            "type": "active",
            "id": f"pair_{i}_rom"
        })
        # English
        en_data.append({
            "text": en,
            "lang": "en_trans",
            "type": "active",
            "id": f"pair_{i}_en"
        })
        
    with open("data/topology_control_full/ja_active.json", "w", encoding="utf-8") as f:
        json.dump(ja_data, f, indent=2, ensure_ascii=False)
        
    with open("data/topology_control_full/ja_romaji_active.json", "w", encoding="utf-8") as f:
        json.dump(romaji_data, f, indent=2, ensure_ascii=False)
        
    with open("data/topology_control_full/en_trans_active.json", "w", encoding="utf-8") as f:
        json.dump(en_data, f, indent=2, ensure_ascii=False)
        
    print(f"Created data/topology_control_full/ja_active.json ({len(ja_data)} items)")
    print(f"Created data/topology_control_full/ja_romaji_active.json ({len(romaji_data)} items)")
    print(f"Created data/topology_control_full/en_trans_active.json ({len(en_data)} items)")

if __name__ == "__main__":
    create_full_controls()
