// Unit tests for KokoroG2P (Swift Testing). Expected sentence strings were produced by misaki
// (`en.G2P`, spaCy en_core_web_sm, no fallback, unk="❓") for American and British English.
import Foundation
import Testing
@testable import KokoroG2P

let misakiCases: [(String, String, String)] = [
    ("The quick brown fox jumps over the lazy dog.",
     "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.",
     "ðə kwˈɪk bɹˈWn fˈɒks ʤˈʌmps ˈQvə ðə lˈAzi dˈɒɡ."),
    ("I read the book last week, and now I read a chapter every night.",
     "ˌI ɹˈɛd ðə bˈʊk lˈæst wˈik, ænd nˈW ˌI ɹˈɛd ɐ ʧˈæptəɹ ˈɛvəɹi nˈIt.",
     "ˌI ɹˈɛd ðə bˈʊk lˈɑːst wˈiːk, and nˈW ˌI ɹˈɛd ɐ ʧˈaptə ˈɛvəɹi nˈIt."),
    ("They live in a small house near the river.",
     "ðˌA lˈɪv ɪn ɐ smˈɔl hˈWs nˌɪɹ ðə ɹˈɪvəɹ.",
     "ðˌA lˈɪv ɪn ɐ smˈɔːl hˈWs nˌɪə ðə ɹˈɪvə."),
    ("I used to play tennis, but I haven't used my racket in years.",
     "ˌI jˈust tə plˈA tˈɛnəs, bˌʌt ˌI hˈævᵊnt jˈuzd mI ɹˈækət ɪn jˈɪɹz.",
     "ˌI jˈuːst tə plˈA tˈɛnɪs, bˌʌt ˌI hˈavᵊnt jˈuːzd mI ɹˈakɪt ɪn jˈɪəz."),
    ("Apple's new iPhone costs $999, while the Pro model starts at $1,199.",
     "ˈæpᵊlz nˈu ˈIfˌOn kˈɔsts nˈIn hˈʌndɹəd nˈIndi nˈIn dˈɑləɹz, wˌIl ðə pɹˈO mˈɑdᵊl stˈɑɹts æt wˈʌn θˈWzᵊnd wˈʌn hˈʌndɹəd nˈIndi nˈIn dˈɑləɹz.",
     "ˈapᵊlz njˈuː ˈIfˌQn kˈɒsts nˈIn hˈʌndɹəd nˈInti nˈIn dˈɒləz, wˌIl ðə pɹˈQ mˈɒdᵊl stˈɑːts at wˈʌn θˈWzᵊnd wˈʌn hˈʌndɹəd nˈInti nˈIn dˈɒləz."),
    ("He was born on July 4, 1976, in Philadelphia.",
     "hˌi wʌz bˈɔɹn ˌɔn ʤəlˈI fˈɔɹ, nˌIntˈin sˈɛvənti sˈɪks, ɪn fˌɪlədˈɛlfiə.",
     "hˌiː wɒz bˈɔːn ˌɒn ʤʊlˈI fˈɔː, nˌIntˈiːn sˈɛvnti sˈɪks, ɪn fˌɪlədˈɛlfɪə."),
    ("The price went from £20 to £35.50 in just two years.",
     "ðə pɹˈIs wˈɛnt fɹʌm twˈɛnti pˈWndz tə θˈɜɹTi fˈIv pˈWndz ænd fˈɪfti pˈɛns ɪn ʤˈʌst tˈu jˈɪɹz.",
     "ðə pɹˈIs wˈɛnt fɹɒm twˈɛnti pˈWndz tə θˈɜːti fˈIv pˈWndz and fˈɪfti pˈɛns ɪn ʤˈʌst tˈuː jˈɪəz."),
    ("My phone number is 555-0123, call me after 6.",
     "mˌI fˈOn nˈʌmbəɹ ɪz fˈIv hˈʌndɹəd fˈɪfti fˈIvwˌʌn twˌɛnti θɹˌi, kˈɔl mˌi ˈæftəɹ sˈɪks.",
     "mˌI fˈQn nˈʌmbə ɪz fˈIv hˈʌndɹəd fˈɪfti fˈIvwˌʌn twˌɛnti θɹˌiː, kˈɔːl mˌiː ˈɑːftə sˈɪks."),
    ("The [Kokoro](/kˈOkəɹO/) model sounds natural.",
     "ðə kˈOkəɹO mˈɑdᵊl sˈWndz nˈæʧəɹᵊl.",
     "ðə kˈOkəɹO mˈɒdᵊl sˈWndz nˈaʧəɹəl."),
    ("The NASA mission was a success, according to the FBI and the CIA.",
     "ðə nˈæsə mˈɪʃən wʌz ɐ səksˈɛs, əkˈɔɹdɪŋ tə ði ˌɛfbˌiˈI ænd ðə sˌiˌIˈA.",
     "ðə nˈasə mˈɪʃᵊn wɒz ɐ səksˈɛs, əkˈɔːdɪŋ tə ði ˌɛfbˌiːˈI and ðə sˌiːˌIˈA."),
    ("She said, \"I'll be there at 5 p.m., don't worry.\"",
     "ʃˌi sˈɛd, “ˌIl bi ðɛɹ æt fˈIv pˌiˈɛm, dˈOnt wˈɜɹi.”",
     "ʃˌiː sˈɛd, “ˌIl biː ðɛː at fˈIv pˌiːˈɛm, dˈQnt wˈʌɹi.”"),
    ("It's a well-known fact that state-of-the-art equipment is expensive.",
     "ˌɪts ɐ wˌɛlnˈOn fˈækt ðæt stˈAtʌvðiˈɑɹt əkwˈɪpmənt ɪz ɪkspˈɛnsɪv.",
     "ˌɪts ɐ wˌɛlnˈQn fˈakt ðat stˈAtɒvðiˈɑːt ɪkwˈɪpmᵊnt ɪz ɪkspˈɛnsɪv."),
    ("The dogs' owners met at the park.",
     "ðə dˈɔɡz ˈOnəɹz mˈɛt æt ðə pˈɑɹk.",
     "ðə dˈɒɡz ˈQnəz mˈɛt at ðə pˈɑːk."),
    ("Dr. Johnson met with Mr. and Mrs. Smith on Elm St. yesterday.",
     "dˈɑktəɹ ʤˈɑnsᵊn mˈɛt wɪð mˈɪstəɹ ænd mˈɪsɪz smˈɪθ ˌɔn ˈɛlm ❓ jˈɛstəɹdˌA.",
     "dˈɒktə ʤˈɒnsᵊn mˈɛt wɪð mˈɪstə and mˈɪsɪz smˈɪθ ˌɒn ˈɛlm ❓ jˈɛstədA."),
    ("The 21st century has brought rapid technological change.",
     "ðə twˈɛnti fˈɜɹst sˈɛnʧəɹi hæz bɹˈɔt ɹˈæpəd tˌɛknəlˈɑʤəkᵊl ʧˈAnʤ.",
     "ðə twˈɛnti fˈɜːst sˈɛnʧʊɹi haz bɹˈɔːt ɹˈapɪd tˌɛknəlˈɒʤɪkᵊl ʧˈAnʤ."),
    ("Water boils at 100 degrees Celsius.",
     "wˈɔTəɹ bˈYlz æt wˈʌn hˈʌndɹəd dəɡɹˈiz sˈɛlsiəs.",
     "wˈɔːtə bˈYlz at wˈʌn hˈʌndɹəd dɪɡɹˈiːz sˈɛlsɪəs."),
]

enum Shared {
    static let us = try! EnglishG2P(british: false, unknownMarker: "❓", useFallback: false)
    static let gb = try! EnglishG2P(british: true, unknownMarker: "❓", useFallback: false)
    static let usFallback = try! EnglishG2P(british: false)
    static let gbFallback = try! EnglishG2P(british: true)
}

// Instances are single-threaded and shared here, so all suites using them are serialized.
@Suite(.serialized) struct KokoroG2PTests {
    func word(_ g: EnglishG2P, _ text: String, _ w: String) -> String? {
        g.phonemize(text).tokens.first(where: { $0.text == w })?.phonemes
    }

    @Test func americanMatchesMisaki() {
        for (text, us, _) in misakiCases {
            #expect(Shared.us.phonemize(text).phonemes == us, "\(text)")
        }
    }

    @Test func britishMatchesMisaki() {
        for (text, _, gb) in misakiCases {
            #expect(Shared.gb.phonemize(text).phonemes == gb, "\(text)")
        }
    }

    @Test func homographsFollowTags() {
        let g = Shared.us
        #expect(word(g, "Please read the instructions.", "read") == "ɹˈid")
        #expect(word(g, "She read the letter twice.", "read") == "ɹˈɛd")
        #expect(word(g, "They live here.", "live") == "lˈɪv")
        #expect(word(g, "He used to run.", "used") == "jˈust")
        #expect(word(g, "The record was broken.", "record") == "ɹˈɛkəɹd")
        #expect(word(g, "They will record it.", "record") == "ɹəkˈɔɹd")
    }

    @Test func functionWordContext() {
        let g = Shared.us
        #expect(word(g, "the apple", "the") == "ði")
        #expect(word(g, "the pear", "the") == "ðə")
        #expect(word(g, "go to it", "to") == "tʊ")
        #expect(word(g, "go to bed", "to") == "tə")
        #expect(word(g, "a dog", "a") == "ɐ")
    }

    @Test func numbersCurrencyAndOrdinals() {
        let g = Shared.us
        #expect(g.phonemize("It costs $5.50.").phonemes == "ˌɪt kˈɔsts fˈIv dˈɑləɹz ænd fˈɪfti sˈɛnts.")
        #expect(word(g, "In 1984 it rained.", "1984") == "nˌIntˈin ˈATi fˈɔɹ")
        #expect(word(g, "The 3rd one.", "3rd") == "θˈɜɹd")
        #expect(word(g, "Pi is 3.14 roughly.", "3.14") == "θɹˈi pYnt wˈʌn fˈɔɹ")
        #expect(word(g, "About 50 % of it.", "%") == "pəɹsˈɛnt")
    }

    @Test func explicitPhonemesViaLinkSyntax() {
        let (ps, toks) = Shared.us.phonemize("Say [Kokoro](/kˈOkəɹO/) now.")
        #expect(ps.contains("kˈOkəɹO"))
        #expect(toks.contains(where: { $0.text == "Kokoro" && $0.rating == 5 }))
    }

    @Test func acronymsAreSpelled() {
        #expect(word(Shared.us, "The FBI said so.", "FBI") == "ˌɛfbˌiˈI")
    }

    @Test func unknownWordsWithoutFallback() {
        let (ps, toks) = Shared.us.phonemize("The zorblaxian fleet arrived.")
        #expect(ps.contains("❓"))
        #expect(toks.first(where: { $0.text == "zorblaxian" })?.phonemes == nil)
    }

    @Test func fallbackProducesKokoroSymbols() {
        let usVocab = Set("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔTɐ")
        let gbVocab = Set("AIQWYabdfhijklmnpstuvwzðŋɑɒɔəɛɜɡɪɹʃʊʌʒʤʧˈˌːθᵊɐ")
        for w in ["zorblaxian", "Kowalski", "Spotify", "frobnicate", "Brockhurst", "Nakamura"] {
            let t = Shared.usFallback.phonemize(w).tokens[0]
            #expect(t.isFallback, "\(w)")
            let p = t.phonemes ?? ""
            #expect(!p.isEmpty && p.allSatisfy { usVocab.contains($0) || $0 == " " }, "\(w): \(p)")
            let pg = Shared.gbFallback.phonemize(w).tokens[0].phonemes ?? ""
            #expect(!pg.isEmpty && pg.allSatisfy { gbVocab.contains($0) || $0 == " " }, "\(w): \(pg)")
        }
    }

    @Test func fallbackSpellsInitialismsAndUsesMorphology() {
        let g = Shared.usFallback
        #expect(g.fallbackPhonemes(for: "XKCD") == "ˌɛkskˌAsˌidˈi")  // letter names from the lexicon
        #expect(g.fallbackPhonemes(for: "unfriendable")?.hasPrefix("ˌʌn") == true)
        #expect(g.fallbackPhonemes(for: "bioshock") != nil)
    }

    @Test func pipelineJoinMatchesKPipeline() {
        let (_, toks) = Shared.usFallback.phonemize("Hello,  world!")
        #expect(EnglishG2P.pipelinePhonemes(toks) == "həlˈO, wˈɜɹld!")
    }

    @Test func secondInstanceReusesCachedLexicons() throws {
        _ = Shared.usFallback
        let t0 = Date()
        _ = try EnglishG2P(british: false)
        #expect(Date().timeIntervalSince(t0) < 0.02)
    }

    @Test func numberWordsLikeNum2words() {
        #expect(NumberWords.cardinal(123) == "one hundred and twenty-three")
        #expect(NumberWords.cardinal(1100) == "one thousand, one hundred")
        #expect(NumberWords.cardinal(1001) == "one thousand and one")
        #expect(NumberWords.cardinal(1_234_567) == "one million, two hundred and thirty-four thousand, five hundred and sixty-seven")
        #expect(NumberWords.cardinal(digits: "2000050") == "two million and fifty")
        #expect(NumberWords.ordinal(digits: "21") == "twenty-first")
        #expect(NumberWords.ordinal(digits: "100") == "one hundredth")
        #expect(NumberWords.year(1905) == "nineteen oh-five")
        #expect(NumberWords.year(2005) == "two thousand and five")
        #expect(NumberWords.year(2019) == "twenty nineteen")
        #expect(NumberWords.year(1900) == "nineteen hundred")
        #expect(NumberWords.float("10.05") == "ten point zero five")
        #expect(NumberWords.float("2.50") == "two point five")
    }
}
