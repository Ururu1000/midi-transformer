from musiclm.inference.prompts import parse_prompt

COMPOSERS = [
    "Frédéric Chopin",
    "Johann Sebastian Bach",
    "Ludwig van Beethoven",
    "Franz Liszt",
    "Claude Debussy",
    "Sergei Rachmaninoff",
    "Wolfgang Amadeus Mozart",
]


class TestComposerMatching:
    def test_chopin_russian_genitive(self):
        parsed = parse_prompt("сгенерируй мне шопена", COMPOSERS)
        assert parsed.composer == "Frédéric Chopin"

    def test_chopin_latin(self):
        parsed = parse_prompt("generate me some chopin", COMPOSERS)
        assert parsed.composer == "Frédéric Chopin"

    def test_bach_russian(self):
        parsed = parse_prompt("сгенерируй баха", COMPOSERS)
        assert parsed.composer == "Johann Sebastian Bach"

    def test_beethoven_russian(self):
        parsed = parse_prompt("хочу бетховена", COMPOSERS)
        assert parsed.composer == "Ludwig van Beethoven"

    def test_unknown_keeps_none(self):
        parsed = parse_prompt("сгенерируй что-нибудь красивое", COMPOSERS)
        assert parsed.composer is None
        assert parsed.length is None

    def test_empty_prompt(self):
        parsed = parse_prompt("   ", COMPOSERS)
        assert parsed.composer is None
        assert parsed.length is None
        assert parsed.temperature is None


class TestLengthMatching:
    def test_long_keyword(self):
        parsed = parse_prompt("сгенерируй шопена, подлиннее", COMPOSERS)
        assert parsed.composer == "Frédéric Chopin"
        assert parsed.length == 2048

    def test_short_keyword(self):
        parsed = parse_prompt("короткий бах", COMPOSERS)
        assert parsed.length == 512

    def test_explicit_tokens(self):
        parsed = parse_prompt("дебюсси на 1500 токенов", COMPOSERS)
        assert parsed.composer == "Claude Debussy"
        assert parsed.length == 1500

    def test_explicit_clamped(self):
        parsed = parse_prompt("моцарт на 9999 токенов", COMPOSERS)
        assert parsed.length == 4096


class TestTemperatureMatching:
    def test_experimental(self):
        parsed = parse_prompt("странный экспериментальный лист", COMPOSERS)
        assert parsed.composer == "Franz Liszt"
        assert parsed.temperature == 1.05

    def test_strict(self):
        parsed = parse_prompt("строгий точный бах", COMPOSERS)
        assert parsed.temperature == 0.85

    def test_default_none(self):
        parsed = parse_prompt("сгенерируй шопена", COMPOSERS)
        assert parsed.temperature is None


class TestEnglish:
    def test_generate_chopin(self):
        parsed = parse_prompt("generate me some Chopin", COMPOSERS)
        assert parsed.composer == "Frédéric Chopin"

    def test_compose_in_style_of(self):
        parsed = parse_prompt("compose something in the style of Bach", COMPOSERS)
        assert parsed.composer == "Johann Sebastian Bach"

    def test_long_piece(self):
        parsed = parse_prompt("generate a long Beethoven piece", COMPOSERS)
        assert parsed.composer == "Ludwig van Beethoven"
        assert parsed.length == 2048

    def test_short_piece(self):
        parsed = parse_prompt("a short Mozart piece", COMPOSERS)
        assert parsed.composer == "Wolfgang Amadeus Mozart"
        assert parsed.length == 512

    def test_two_minutes(self):
        parsed = parse_prompt("a two minutes Debussy piece", COMPOSERS)
        assert parsed.composer == "Claude Debussy"
        assert parsed.length == 2048

    def test_explicit_tokens_en(self):
        parsed = parse_prompt("Liszt, 1500 tokens", COMPOSERS)
        assert parsed.composer == "Franz Liszt"
        assert parsed.length == 1500

    def test_creative(self):
        parsed = parse_prompt("something weird and creative by Rachmaninoff", COMPOSERS)
        assert parsed.composer == "Sergei Rachmaninoff"
        assert parsed.temperature == 1.05

    def test_strict_en(self):
        parsed = parse_prompt("a strict, faithful Bach", COMPOSERS)
        assert parsed.temperature == 0.85

    def test_unknown_en(self):
        parsed = parse_prompt("generate something beautiful", COMPOSERS)
        assert parsed.composer is None
        assert parsed.length is None
        assert parsed.temperature is None
