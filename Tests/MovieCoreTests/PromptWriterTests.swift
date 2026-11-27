import Testing
@testable import MovieCore

@Suite struct PromptWriterTests {
    private func spec() -> PromptSpec {
        var s = PromptSpec()
        s.medium = "35mm film photo"
        s.subject = "a 40-year-old adult man"
        s.pronoun = .he
        s.shot = "medium shot"
        s.action = "Reading a newspaper"
        s.details = [PromptDetail("grey stubble")]
        s.body = [PromptDetail(tag: "stocky build", text: "a stocky, broad build", weight: 1.2)]
        s.outfit = [PromptDetail(tag: "tweed jacket", text: "a worn brown tweed jacket", weight: 1.2)]
        s.place = "a station platform"
        s.time = ["early morning", "light rain"]
        s.camera = ["eye level", "50mm lens"]
        s.lighting = ["soft overcast light"]
        s.mood = ["quiet", "melancholic"]
        s.palette = ["muted teal and amber"]
        s.avoid = ["blurry", "watermark"]
        s.without = ["no other people"]
        return s
    }

    @Test func tagsPutTheSubjectFirstAndWeighTheLook() {
        let w = PromptWriter.write(spec(), for: .sdxl)
        #expect(w.prompt.hasPrefix("35mm film photo, a 40-year-old adult man, medium shot, reading a newspaper, (stocky build:1.2), (tweed jacket:1.2), grey stubble, a station platform"))
        #expect(w.negative == "blurry, watermark")
        // Anima reads tags but not A1111 weights; Pony puts its score tags first and in the negative.
        #expect(PromptWriter.write(spec(), for: .anima).prompt.contains(", stocky build, tweed jacket,"))
        let pony = PromptWriter.write(spec(), for: .sdxlPony)
        #expect(pony.prompt.hasPrefix("score_9, score_8_up, score_7_up, 35mm film photo") && pony.negative.hasPrefix("score_6"))
    }

    @Test func everyDialectLeadsWithLoRATriggerWords() {
        var s = spec()
        s.triggerWords = ["zqmia_3f9", "toned abdomen"]
        #expect(PromptWriter.write(s, for: .sdxl).prompt.hasPrefix("zqmia_3f9, toned abdomen, 35mm film photo"))
        for m in [PromptModel.chroma, .minimaxImage, .minimaxVideo, .wan] {
            #expect(PromptWriter.write(s, for: m).prompt.hasPrefix("zqmia_3f9, toned abdomen. "), "\(m)")
        }
        s.triggerWords = []
        #expect(PromptWriter.write(s, for: .minimaxImage).prompt.hasPrefix("A 35mm film photo"))
    }

    @Test func sentencesDescribeWhatIsVisible() {
        let w = PromptWriter.write(spec(), for: .chroma)
        #expect(w.prompt.hasPrefix("A 35mm film photo, medium shot, of a 40-year-old adult man reading a newspaper."))
        #expect(w.prompt.contains("He has grey stubble. He has a stocky, broad build. He wears a worn brown tweed jacket."))
        #expect(w.prompt.contains("At a station platform, early morning and light rain.") && w.prompt.contains("The mood is quiet and melancholic."))
        #expect(!w.prompt.contains("(") && w.negative == "blurry, watermark" && !w.prompt.contains("No other people"))
        // FLUX takes no negative: the exclusions become sentences.
        let flux = PromptWriter.write(spec(), for: .flux)
        #expect(flux.negative == "" && flux.prompt.hasSuffix("No other people."))
    }

    @Test func narrativeFollowsMiniMaxsGuide() {
        var s = spec()
        s.reference = .faceAndHair
        let w = PromptWriter.write(s, for: .minimaxImage)
        #expect(w.negative == "")
        #expect(w.prompt.hasPrefix("<Picture 1> shows the face of a 40-year-old adult man. A 35mm film photo, medium shot, of that same 40-year-old adult man reading a newspaper."))
        #expect(w.prompt.contains("His face and hair are kept exactly as in <Picture 1>; his body, clothes, pose and background are as described here."))
        #expect(w.prompt.hasSuffix("The image contains no text, letters or numbers. No other people."))
        s.text = "LAST TRAIN"
        s.reference = .none
        let t = PromptWriter.write(s, for: .minimaxImage).prompt
        #expect(t.contains("The text \"LAST TRAIN\" appears in the picture, spelled exactly.") && !t.contains("no text"))
        #expect(!t.contains("<Picture 1>"))
    }

    @Test func motionDescribesTheShotThenTheActionAndSound() {
        var s = spec()
        s.camera = ["slow dolly in"]
        s.sound = ["rain on the roof", "a distant train horn"]
        let ltx = PromptWriter.write(s, for: .ltx23).prompt
        #expect(ltx.hasPrefix("A 35mm film photo, a medium shot, slow dolly in. A 40-year-old adult man reading a newspaper."))
        #expect(ltx.hasSuffix("Sound: rain on the roof and a distant train horn."))
        #expect(!PromptWriter.write(s, for: .wan).prompt.contains("Sound:"))
        #expect(PromptWriter.write(s, for: .wan).negative == "blurry, watermark")
        #expect(PromptWriter.write(s, for: .minimaxVideo).negative == "")
    }

    @Test func captionsThatStartWithANounFollowAComma() {
        var s = PromptSpec()
        s.medium = "candid smartphone photo"
        s.subject = "a 25-year-old adult woman"
        s.action = "Selfie on a rooftop"
        #expect(PromptWriter.write(s, for: .chroma).prompt == "A candid smartphone photo of a 25-year-old adult woman, selfie on a rooftop.")
        s.action = "sitting on a rooftop"
        #expect(PromptWriter.write(s, for: .chroma).prompt == "A candid smartphone photo of a 25-year-old adult woman sitting on a rooftop.")
    }

    @Test func sdxlFlavourFromTheFineTunesBaseModel() {
        #expect(PromptModel.sdxl(baseModel: "Pony") == .sdxlPony)
        #expect(PromptModel.sdxl(baseModel: "Illustrious") == .sdxlIllustrious)
        #expect(PromptModel.sdxl(baseModel: "NoobAI") == .sdxlIllustrious)
        #expect(PromptModel.sdxl(baseModel: "SDXL 1.0") == .sdxl && PromptModel.sdxl(baseModel: nil) == .sdxl)
        #expect(PromptModel.allCases.filter(\.takesWeights) == [.sdxl, .sdxlPony, .sdxlIllustrious])
        #expect(PromptModel.allCases.filter(\.hasSound) == [.ltx23, .minimaxVideo])
    }
}

@Suite struct PromptCoverageTests {
    private func person(_ c: PromptSpec.Coverage) -> PromptSpec {
        var s = PromptSpec()
        s.subject = "a 30-year-old adult woman"
        s.pronoun = .she
        s.outfit = [PromptDetail("a red dress", weight: 1.2)]
        s.coverage = c
        return s
    }

    @Test func coverageReplacesTheOutfitInEveryDialect() {
        let tags = PromptWriter.write(person(.nude), for: .sdxl).prompt
        #expect(tags.contains("(fully nude, naked:1.2)") && !tags.contains("red dress"))
        #expect(tags.contains("adult body with mature proportions"))  // tags drop the article
        let chroma = PromptWriter.write(person(.nude), for: .chroma).prompt
        #expect(chroma.contains("She is fully nude.") && !chroma.contains("wears"))
        let h3 = PromptWriter.write(person(.topless), for: .minimaxImage).prompt
        #expect(h3.contains("She is topless") && h3.contains("The person is clearly a grown adult."))
        let swim = PromptWriter.write(person(.swimwear), for: .chroma).prompt
        #expect(swim.contains("She wears a swimsuit.") && !swim.contains(PromptSpec.adultBody))
        let dressed = PromptWriter.write(person(.dressed), for: .chroma).prompt
        #expect(dressed.contains("She wears a red dress."))
        var they = person(.impliedNude)
        they.pronoun = .they
        #expect(PromptWriter.write(they, for: .flux).prompt.contains("They are nude but covered by a bedsheet"))
    }
}
