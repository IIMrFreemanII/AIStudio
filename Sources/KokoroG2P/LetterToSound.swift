// Rule-based pronunciation of out-of-vocabulary words (replaces misaki's espeak/neural fallback).
//
// Pipeline for one word:
//   1. letters only → spell short all-caps words / vowelless words letter by letter;
//   2. morphological decomposition with the lexicon: known prefix + word, word + suffix, compounds;
//   3. otherwise English letter-to-sound rules in the style of the public-domain NRL rules
//      (Elovitz, Johnson, McHugh & Shore, "Letter-to-Sound Rules for Automatic Translation of English
//      Text to Phonetics", NRL Report 7948, 1976), rewritten and extended here, producing ARPAbet-like
//      phones; then stress assignment (suffix rules + a Latin-style weight rule), vowel reduction,
//      American flapping, and rendering to misaki's US or GB phoneme alphabet.
import Foundation

final class LetterToSound {
    let lexicon: Lexicon
    let british: Bool
    /// When set, this word (lowercased) is treated as unknown during decomposition (evaluation only).
    var excludedWord: String? = nil

    init(lexicon: Lexicon, british: Bool) {
        self.lexicon = lexicon
        self.british = british
    }

    // MARK: Entry point

    func phonemize(_ raw: String) -> String? {
        // Fold accents (café → cafe) and keep letters, apostrophes and separators.
        let folded = raw.folding(options: [.diacriticInsensitive, .widthInsensitive], locale: Locale(identifier: "en_US"))
            .replacingOccurrences(of: "’", with: "'").replacingOccurrences(of: "‘", with: "'")
        var parts: [String] = []
        var cur = ""
        for ch in folded {
            if ch.isLetter && ch.isASCII || ch == "'" { cur.append(ch) } else if ch.isNumber {
                if !cur.isEmpty { parts.append(cur); cur = "" }
                parts.append(String(ch))
            } else {
                if !cur.isEmpty { parts.append(cur); cur = "" }
            }
        }
        if !cur.isEmpty { parts.append(cur) }
        var out: [String] = []
        var i = 0
        while i < parts.count {
            let p = parts[i]
            if p.first!.isNumber {
                // A digit run inside an unknown token: read the digits one by one.
                var digits = p
                while i + 1 < parts.count, parts[i + 1].first!.isNumber { i += 1; digits += parts[i] }
                let words = digits.compactMap { $0.wholeNumberValue }.map { NumberWords.ones[$0] }
                let ps = words.compactMap { lexicon.lookup($0, nil, nil, nil).0 }
                if !ps.isEmpty { out.append(ps.joined(separator: " ")) }
            } else if let ps = word(p.trimmingCharacters(in: CharacterSet(charactersIn: "'"))) {
                out.append(ps)
            }
            i += 1
        }
        return out.isEmpty ? nil : out.joined(separator: " ")
    }

    /// One alphabetic word (may contain apostrophes).
    func word(_ w: String) -> String? {
        guard !w.isEmpty else { return nil }
        let letters = w.filter { $0 != "'" }
        let lower = letters.lowercased()
        let vowelCount = lower.filter { "aeiouy".contains($0) }.count
        // Spell initialisms: short all-caps words, or words without any vowel letter.
        if letters.count > 1 && letters == letters.uppercased() && (letters.count <= 4 || !isPronounceable(lower)) {
            return spell(letters)
        }
        if vowelCount == 0 || (letters.count == 1) { return spell(letters) }
        if let d = decompose(lower) { return d }
        return rules(lower)
    }

    func spell(_ w: String) -> String? {
        let (ps, _) = lexicon.getNNP(w)
        return ps
    }

    /// Rough check that an all-caps word can be read as a word (NASA, UNESCO) rather than spelled.
    func isPronounceable(_ w: String) -> Bool {
        let v = Set("aeiouy")
        var run = 0, maxRun = 0
        for c in w { if v.contains(c) { run = 0 } else { run += 1; maxRun = max(maxRun, run) } }
        let vowels = w.filter { v.contains($0) }.count
        return maxRun <= 3 && vowels * 3 >= w.count && w.count >= 5
    }

    // MARK: Morphological decomposition

    private func known(_ w: String) -> String? {
        if w == excludedWord { return nil }
        guard w.count >= 2 else { return nil }
        let (ps, _) = lexicon.getWord(w, nil, nil, TokenContext())
        guard let ps, !ps.isEmpty else { return nil }
        return ps
    }

    private func knownGold(_ w: String) -> String? {
        if w == excludedWord { return nil }
        guard lexicon.golds[w] != nil else { return nil }
        return known(w)
    }

    static let prefixes: [(String, String, String)] = [  // spelling, US, GB
        ("counter", "kˌWntəɹ", "kˌWntə"), ("under", "ˌʌndəɹ", "ˌʌndə"), ("over", "ˌOvəɹ", "ˌQvə"),
        ("super", "sˌupəɹ", "sˌuːpə"), ("inter", "ˌɪntəɹ", "ˌɪntə"), ("hyper", "hˌIpəɹ", "hˌIpə"),
        ("cyber", "sˌIbəɹ", "sˌIbə"), ("ultra", "ˌʌltɹə", "ˌʌltɹə"), ("multi", "mˌʌlti", "mˌʌlti"),
        ("micro", "mˌIkɹO", "mˌIkɹQ"), ("macro", "mˌækɹO", "mˌakɹQ"), ("mega", "mˌɛɡə", "mˌɛɡə"),
        ("anti", "ˌænti", "ˌanti"), ("semi", "sˌɛmi", "sˌɛmi"), ("mini", "mˌɪni", "mˌɪni"),
        ("auto", "ˌɔɾO", "ˌɔːtQ"), ("tele", "tˌɛlə", "tˌɛlɪ"), ("trans", "tɹˌænz", "tɹˌanz"),
        ("post", "pˌOst", "pˌQst"), ("fore", "fˌɔɹ", "fˌɔː"), ("out", "ˌWt", "ˌWt"), ("self", "sˌɛlf", "sˌɛlf"),
        ("non", "nˌɑn", "nˌɒn"), ("pre", "pɹˌi", "pɹˌiː"), ("dis", "dˌɪs", "dˌɪs"), ("mis", "mˌɪs", "mˌɪs"),
        ("sub", "sˌʌb", "sˌʌb"), ("un", "ˌʌn", "ˌʌn"), ("re", "ɹˌi", "ɹˌiː"), ("bio", "bˌIO", "bˌIQ"),
        ("eco", "ˌikO", "ˌiːkQ"), ("geo", "ʤˌiO", "ʤˌiːQ"), ("neuro", "nˌʊɹO", "njˌʊəɹQ"), ("co", "kˌO", "kˌQ"),
        ("de", "dˌi", "dˌiː"), ("up", "ˌʌp", "ˌʌp"), ("down", "dˌWn", "dˌWn"), ("back", "bˌæk", "bˌak"),
    ]

    static let suffixes: [(String, String, String)] = [  // spelling, US, GB (attached to the stem)
        ("ness", "nəs", "nəs"), ("less", "ləs", "ləs"), ("ful", "fəl", "fəl"), ("ment", "mənt", "mənt"),
        ("ly", "li", "li"), ("ship", "ʃˌɪp", "ʃˌɪp"), ("hood", "hˌʊd", "hˌʊd"), ("like", "lˌIk", "lˌIk"),
        ("wise", "wˌIz", "wˌIz"), ("dom", "dəm", "dəm"), ("ism", "ˌɪzəm", "ˌɪzəm"), ("ist", "ɪst", "ɪst"),
        ("ish", "ɪʃ", "ɪʃ"), ("able", "əbᵊl", "əbᵊl"), ("er", "əɹ", "ə"), ("ers", "əɹz", "əz"), ("y", "i", "i"),
        ("ize", "ˌIz", "ˌIz"), ("ise", "ˌIz", "ˌIz"), ("ization", "əzˈAʃən", "IzˈAʃən"),
        ("land", "lənd", "lənd"), ("ville", "vˌɪl", "vˌɪl"), ("burg", "bˌɜɹɡ", "bˌɜːɡ"), ("ton", "tən", "tən"),
        ("field", "fˌild", "fˌiːld"), ("berg", "bˌɜɹɡ", "bˌɜːɡ"), ("son", "sən", "sən"), ("stein", "stˌIn", "stˌIn"),
    ]

    private func demote(_ ps: String) -> String {
        ps.replacingOccurrences(of: "ˈ", with: "ˌ")
    }

    func decompose(_ w: String) -> String? {
        // Stem + suffix (with e-restoration and y→i).
        for (suf, us, gb) in LetterToSound.suffixes where w.hasSuffix(suf) && w.count >= suf.count + 3 {
            let stem = String(w.dropLast(suf.count))
            var cands = [stem]
            if stem.hasSuffix("i") { cands.append(String(stem.dropLast()) + "y") }
            if !stem.hasSuffix("e") { cands.append(stem + "e") }
            if stem.count >= 3, let l = stem.last, stem.dropLast().last == l, !"aeiou".contains(l) { cands.append(String(stem.dropLast())) }
            for c in cands {
                if let ps = known(c) {
                    var base = ps
                    if c.hasSuffix("y") && stem.hasSuffix("i"), base.hasSuffix("i") { base = String(base.dropLast()) + "i" }
                    let s = british ? gb : us
                    if (suf == "er" || suf == "ers") && c == stem + "e" { /* "make" + "er" */ }
                    return base + s
                }
            }
        }
        // Prefix + known word.
        for (pre, us, gb) in LetterToSound.prefixes where w.hasPrefix(pre) && w.count >= pre.count + 3 {
            let rest = String(w.dropFirst(pre.count))
            if let ps = known(rest) {
                return (british ? gb : us) + ps
            }
        }
        // Compound of two known words (each ≥ 3 letters), the longer first part wins.
        if w.count >= 6 {
            let chars = Array(w)
            for cut in stride(from: chars.count - 3, through: 3, by: -1) {
                let a = String(chars[..<cut]), b = String(chars[cut...])
                guard a.count >= 4 && b.count >= 4 || a.count >= 3 && b.count >= 5, let pa = knownGold(a), let pb = knownGold(b) ?? (b.count >= 4 ? known(b) : nil) else { continue }
                // Avoid splitting off tiny function-word-like parts ("on", "in" are excluded by length).
                return pa + demote(pb)
            }
        }
        return nil
    }

    // MARK: Letter-to-sound rules

    struct Rule {
        let left: [Character]
        let match: [Character]
        let right: [Character]
        let phones: [String]
    }

    /// Rules "LEFT[MATCH]RIGHT=PHONES". Context symbols: `_` word boundary, `#` one or more vowels,
    /// `*` one or more consonants, `.` voiced consonant, `%` suffix (ER E ES ED ING ELY), `&` sibilant,
    /// `@` T S R D L Z N J TH CH SH, `^` one consonant, `+` E I Y, `:` zero or more consonants.
    static let ruleText: [String] = [
        // A
        "_[ARE]_=AA r", "_[AR]O=AX r", "[AR]#=EH r", "^[AS]#=EY s", "[A]WA=AX", "[AW]=AO", "_:[ANY]=EH n IY",
        "[A]^+#=EY", "#:[ALLY]=AX l IY", "_[AL]#=AX l", "[AGAIN]=AX g EH n", "#:[AG]E=IH JH", "[A]^+:#=AE",
        "_:[A]^+_=EY", "[A]^%=EY", "_[ARR]=AX r", "[ARR]=AE r", "_:[AR]_=AA r", "[AR]_=AXR", "[AR]=AA r",
        "[AIR]=EH r", "[AI]=EY", "[AY]=EY", "[AU]=AO", "#:[AL]_=AX l", "#:[ALS]_=AX l z", "[ALK]=AO k",
        "[ALL]#=AE l", "[AL]^=AO l", "_:[ABLE]=EY b AX l", "[ABLE]=AX b AX l", "[ANG]+=EY n JH", "[ATION]=EY SH AX n",
        "[AE]=IY", "[AA]=AA", "[AH]_=AA", "[A]_=AX", "[A]=AE",
        // B
        "_[BE]^#=b IH", "[BEING]=b IY IH NG", "_[BOTH]_=b OW TH", "_[BUS]#=b IH z", "[BUIL]=b IH l",
        "[B]B=", "M[B]_=", "[B]=b",
        // C
        "_[CH]^=k", "^E[CH]=k", "[CHR]=k r", "[CH]=CH", "_S[CI]#=s AY", "[CI]A=SH", "[CI]O=SH", "[CI]EN=SH",
        "[CE]A=SH", "[CC]+=k s", "[C]+=s", "[CK]=k", "[COM]%=k AH m", "[CQU]=k w", "[C]=k",
        // D
        "#:[DED]_=d IX d", ".E[D]_=d", "#:^E[D]_=t", "_[DE]^#=d IH", "_[DO]_=d UW", "_[DOES]=d AH z",
        "_[DOING]=d UW IH NG", "_[DOW]=d AW", "[DU]A=JH UW", "[DG]E=JH", "[D]D=", "[D]=d",
        // E
        "#:[E]_=", "':^[E]_=", "_:[E]_=IY", "#[ED]_=d", "#:[E]D_=", "[EV]ER=EH v", "[E]^%=IY", "[E]RR=EH", "[ERI]#=IY r IY",
        "[ERI]=EH r IH", "#:[ER]#=AXR", "[ER]#=EH r", "[ER]=ER", "_[EVEN]=IY v EH n", "#:[E]W=", "@[EW]=UW",
        "[EW]=y UW", "[E]O=IY", "#:&[ES]_=IX z", "#:[E]S_=", "#:[ELY]_=l IY", "#:[EMENT]=m EH n t",
        "[EFUL]=f UH l", "[EE]=IY", "[EARN]=ER n", "_[EAR]^=ER", "[EAD]=EH d", "#:[EA]_=IY AX", "[EA]SU=EH",
        "[EAU]=OW", "[EA]=IY", "[EIGH]=EY", "[EI]=IY", "_[EYE]=AY", "[EY]_=IY", "[EY]=EY", "@[EU]=UW", "[EU]=y UW", "[E]=EH",
        // F
        "[FUL]=f UH l", "[F]F=", "[F]=f",
        // G
        "[GIV]=g IH v", "_[G]I^=g", "[GE]T=g EH", "SU[GGES]=g JH EH s", "[GG]=g", "_B#[G]=g", "[G]+=JH",
        "[GREAT]=g r EY t", "#[GH]=", "_[GN]=n", "[GN]_=n", "[GUE]_=g", "[G]=g",
        // H
        "_[HAV]=h AE v", "_[HERE]=h IY r", "_[HOUR]=AW AXR", "[HOW]=h AW", "[H]#=h", "[H]=",
        // I
        "[IFY]_=IH f AY", "[IFIE]=IH f AY", "_[IN]=IH n", "_[I]_=AY", "[IN]D=AY n", "[IER]=IY AXR", "#:R[IED]=IY d", "[IED]_=AY d", "[IEN]=IY EH n",
        "[IE]T=AY EH", "_:[I]%=AY", "[I]%=IY", "[IE]=IY", "[I]^+:#=IH", "[I]RR=IH", "[IR]#=AY r", "[IZ]%=AY z",
        "[IS]%=AY z", "[I]D%=AY", "+^[I]^+=IH", "[I]T%=AY", "#:^[I]^+=IH", "[I]^+=AY", "[IR]=ER", "[IGH]=AY",
        "[ILD]=AY l d", "[IGN]_=AY n", "[IGN]^=AY n", "[IGN]%=AY n", "[IQUE]=IY k", "[ILLE]_=IH l", "[I]A=IY", "[I]O=IY", "^[I]_=IY",
        "[I]=IH",
        // J
        "[J]=JH",
        // K
        "_[K]N=", "[K]=k",
        // L
        "[LO]C#=l OW", "^[L]ER=l", "L[L]=", "#:^[L]%=AX l", "[LEAD]=l IY d", "[LE]_=AX l", "[L]=l",
        // M
        "[MOV]=m UW v", "[M]M=", "[M]=m",
        // N
        "E[NG]+=n JH", "[NG]R=NG g", "[NG]#=NG g", "[NGL]%=NG g AX l", "[NG]=NG", "[NK]=NG k", "_[NOW]_=n AW",
        "[N]N=", "[N]=n",
        // O
        "[OF]_=AX v", "[OROUGH]=ER OW", "#:[OR]_=AXR", "#:[ORS]_=AXR z", "[O]RR=OL", "[OR]=AO r", "_[ONE]=w AH n",
        "[OW]=OW", "_[OVER]=OW v AXR", "[OV]=AH v", "[O]^%=OW", "[O]^EN=OW", "[O]^I#=OW", "[OL]D=OW l",
        "[OUGHT]=AO t", "[OUGH]=AH f", "_[OU]=AW", "H[OU]S#=AW", "[OUS]=AX s", "[OUR]=AO r", "[OULD]=UH d",
        "^[OU]^L=AH", "[OUP]=UW p", "[OU]=AW", "[OY]=OY", "[OING]=OW IH NG", "[OI]=OY", "[OOR]=AO r",
        "[OOK]=UH k", "[OOD]=UH d", "[OO]=UW", "[O]E=OW", "[O]_=OW", "[OA]=OW", "_[ONLY]=OW n l IY",
        "_[ONCE]=w AH n s", "[ON'T]=OW n t", "C[O]N=OL", "[O]NG=AO", "_:^[O]N=AH", "I[ON]=AX n", "#:[ON]_=AX n",
        "#^[ON]_=AX n", "[O]ST_=OW", "[OF]^=AO f", "[OTHER]=AH DH AXR", "[OSS]_=AO s", "#:^[OM]=AH m",
        "[O]=OL",
        // P
        "[PH]=f", "[PEOP]=p IY p", "[POW]=p AW", "[PUT]_=p UH t", "[P]P=", "_[PS]=s", "_[PN]=n", "[P]=p",
        // Q
        "[QUAR]=k w AO r", "[QUE]_=k", "[QU]=k w", "[Q]=k",
        // R
        "_[RE]^#=r IY", "[RH]=r", "[R]R=", "[R]=r",
        // S
        "[SH]=SH", "#[SION]=ZH AX n", "[SOME]=s AH m", "#[SUR]#=ZH AXR", "[SUR]#=SH AXR", "#[SU]#=ZH UW",
        "#[SSU]#=SH UW", "#[SED]_=z d", "#[S]#=z", "[SAID]=s EH d", "^[SION]=SH AX n", "[S]S=", ".[S]_=z",
        "#:.E[S]_=z", "#:^##[S]_=z", "#:^#[S]_=s", "U[S]_=s", "_:#[S]_=z", "_[SCH]=s k", "[S]C+=", "#[SM]=z m",
        "#[SN]'=z AX n", "[S]=s",
        // T
        "_[THE]_=DH AX", "[TO]_=t UW", "[THAT]_=DH AE t", "_[THIS]_=DH IH s", "_[THEY]=DH EY", "_[THERE]=DH EH r",
        "[THER]=DH AXR", "[THEIR]=DH EH r", "_[THAN]_=DH AE n", "_[THEM]_=DH EH m", "[THESE]_=DH IY z",
        "_[THEN]=DH EH n", "[THROUGH]=TH r UW", "[THOSE]=DH OW z", "[THOUGH]_=DH OW", "_[THUS]=DH AH s",
        "[TH]=TH", "#:[TED]_=t IX d", "S[TI]#N=CH", "[TI]O=SH", "[TI]A=SH", "[TIEN]=SH AX n", "[TUR]#=CH AXR",
        "[TU]A=CH UW", "_[TWO]=t UW", "[TCH]=CH", "[TZ]=t s", "[T]T=", "[T]=t",
        // U
        "_[UN]I=y UW n", "_[UN]=AH n", "_[UPON]=AX p AO n", "[U]RR=AH", "@[UR]#=UH r", "[UR]#=y UH r", "[UR]=ER",
        "[U]^_=AH", "[U]^^=AH", "[UY]=AY", "_G[U]#=", "G[U]%=", "G[U]#=w", "#N[U]=y UW", "@[U]=UW", "[U]=y UW",
        // V
        "[VIEW]=v y UW", "[V]=v",
        // W
        "_[WERE]=w ER", "[WA]S=w OL", "[WA]T=w OL", "[WHERE]=w EH r", "[WHAT]=w OL t", "[WHOL]=h OW l",
        "[WHO]=h UW", "[WH]=w", "[WAR]=w AO r", "[WOR]^=w ER", "[WR]=r", "[W]=w",
        // X
        "_[X]=z", "[X]=k s",
        // Y
        "[YOUNG]=y AH NG", "_[YOU]=y UW", "_[YES]=y EH s", "_[Y]#=y", "#:^[Y]_=IY", "#:^[Y]I=IY", "_:[Y]_=AY",
        "_:[Y]#=AY", "_:[Y]^+:#=IH", "_:[Y]^#=AY", "#[Y]=y", "[Y]=IH",
        // Z
        "[ZZ]=z", "[ZH]=ZH", "[Z]=z",
        // apostrophe
        "[']=",
    ]

    static let ruleTable: [Character: [Rule]] = {
        var t: [Character: [Rule]] = [:]
        for line in ruleText {
            let eq = line.firstIndex(of: "=")!
            let lhs = line[..<eq], rhs = line[line.index(after: eq)...]
            let lb = lhs.firstIndex(of: "[")!, rb = lhs.firstIndex(of: "]")!
            let left = Array(lhs[..<lb]), match = Array(lhs[lhs.index(after: lb)..<rb]), right = Array(lhs[lhs.index(after: rb)...])
            let phones = rhs.split(separator: " ").map(String.init)
            t[match[0], default: []].append(Rule(left: left, match: match, right: right, phones: phones))
        }
        return t
    }()

    static let vowelLetters: Set<Character> = ["A", "E", "I", "O", "U"]
    static let voiced: Set<Character> = ["B", "D", "V", "G", "J", "L", "M", "N", "R", "W", "Z"]
    static let front: Set<Character> = ["E", "I", "Y"]

    private static func isConsonant(_ c: Character) -> Bool { c.isLetter && !vowelLetters.contains(c) }

    /// Matches a right-context pattern starting at `pos`.
    private static func matchRight(_ pat: [Character], _ w: [Character], _ pos0: Int) -> Bool {
        var pos = pos0
        var k = 0
        while k < pat.count {
            let p = pat[k]
            let c: Character? = pos < w.count ? w[pos] : nil
            switch p {
            case "_": if c != nil && c!.isLetter { return false }
            case "#":
                guard let c, vowelLetters.contains(c) || (c == "Y" && pos > 0) else { return false }
                pos += 1
                while pos < w.count && (vowelLetters.contains(w[pos]) || w[pos] == "Y") { pos += 1 }
                k += 1; continue
            case "*":
                guard let c, isConsonant(c) else { return false }
                pos += 1
                while pos < w.count && isConsonant(w[pos]) { pos += 1 }
                k += 1; continue
            case ":":
                while pos < w.count && isConsonant(w[pos]) { pos += 1 }
                k += 1; continue
            case ".": guard let c, voiced.contains(c) else { return false }
            case "^": guard let c, isConsonant(c) else { return false }
            case "+": guard let c, front.contains(c) else { return false }
            case "&":
                guard let c else { return false }
                if (c == "C" || c == "S") && pos + 1 < w.count && w[pos + 1] == "H" { pos += 2; k += 1; continue }
                guard "SCGZXJ".contains(c) else { return false }
            case "@":
                guard let c else { return false }
                if (c == "T" || c == "C" || c == "S") && pos + 1 < w.count && w[pos + 1] == "H" { pos += 2; k += 1; continue }
                guard "TSRDLZNJ".contains(c) else { return false }
            case "%":
                let rest = String(w[min(pos, w.count)...])
                for suf in ["ING", "ELY", "ER", "ES", "ED", "E"] where rest.hasPrefix(suf) {
                    let end = pos + suf.count
                    if end >= w.count || !w[end].isLetter { pos = end; k += 1; break }
                }
                if k < pat.count && pat[k] == "%" { return false }
                continue
            default: if c != p { return false }
            }
            pos += 1
            k += 1
        }
        return true
    }

    /// Matches a left-context pattern ending just before `pos` (pattern read right to left).
    private static func matchLeft(_ pat: [Character], _ w: [Character], _ pos0: Int) -> Bool {
        var pos = pos0 - 1
        var k = pat.count - 1
        while k >= 0 {
            let p = pat[k]
            let c: Character? = pos >= 0 ? w[pos] : nil
            switch p {
            case "_": if c != nil && c!.isLetter { return false }
            case "#":
                guard let c, vowelLetters.contains(c) else { return false }
                pos -= 1
                while pos >= 0 && vowelLetters.contains(w[pos]) { pos -= 1 }
                k -= 1; continue
            case "*":
                guard let c, isConsonant(c) else { return false }
                pos -= 1
                while pos >= 0 && isConsonant(w[pos]) { pos -= 1 }
                k -= 1; continue
            case ":":
                while pos >= 0 && isConsonant(w[pos]) { pos -= 1 }
                k -= 1; continue
            case ".": guard let c, voiced.contains(c) else { return false }
            case "^": guard let c, isConsonant(c) else { return false }
            case "+": guard let c, front.contains(c) else { return false }
            case "&":
                guard let c else { return false }
                if c == "H" && pos >= 1 && (w[pos - 1] == "C" || w[pos - 1] == "S") { pos -= 2; k -= 1; continue }
                guard "SCGZXJ".contains(c) else { return false }
            case "@":
                guard let c else { return false }
                if c == "H" && pos >= 1 && "TCS".contains(w[pos - 1]) { pos -= 2; k -= 1; continue }
                guard "TSRDLZNJ".contains(c) else { return false }
            default: if c != p { return false }
            }
            pos -= 1
            k -= 1
        }
        return true
    }

    struct Phone { var p: String; var letter: Int }

    func applyRules(_ word: String) -> [Phone] {
        let w = Array(word.uppercased())
        var out: [Phone] = []
        var i = 0
        while i < w.count {
            guard let rules = LetterToSound.ruleTable[w[i]] else { i += 1; continue }
            var matched = false
            for r in rules {
                guard i + r.match.count <= w.count, Array(w[i..<(i + r.match.count)]) == r.match else { continue }
                guard LetterToSound.matchLeft(r.left, w, i), LetterToSound.matchRight(r.right, w, i + r.match.count) else { continue }
                for ph in r.phones { out.append(Phone(p: ph, letter: i)) }
                i += r.match.count
                matched = true
                break
            }
            if !matched { i += 1 }
        }
        return out
    }

    // MARK: Stress, reduction and rendering

    static let vowelPhones: Set<String> = ["AA", "OL", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH", "ER", "EY",
                                           "IH", "IX", "IY", "OW", "OY", "UH", "UW"]
    static let longPhones: Set<String> = ["AA", "AO", "AW", "AY", "EY", "IY", "OW", "OY", "UW"]

    /// Suffixes (spelling) that put the primary stress on the syllable right before them.
    static let preStressSuffixes = ["tion", "sion", "cian", "tial", "cial", "ious", "eous", "uous", "ian", "ial",
                                    "ient", "ience", "iency", "ium", "ity", "ify", "ical", "ically", "ic", "ics",
                                    "ior", "ual", "ular", "ulous", "itive", "itude", "ia", "ible", "ishes", "ish"]
    /// Suffixes that carry the primary stress themselves.
    static let stressedSuffixes = ["eer", "ee", "ese", "ette", "esque", "ique", "oon", "aire", "ade", "eau"]
    static let unstressedPrefixes = ["be", "de", "re", "con", "com", "pre", "pro", "ex", "in", "im", "dis", "mis",
                                     "a", "e", "ab", "ad", "ob", "per", "sub", "sur", "trans"]

    /// Stress-neutral suffixes (spelling): stress is assigned on the stem before them.
    static let neutralSuffixes = ["ably", "ibly", "able", "ible", "nesses", "ness", "lessly", "less", "ments", "ment",
                                  "fully", "ful", "ings", "ing", "edly", "ed", "ers", "er", "est", "ly", "es", "s",
                                  "ists", "ist", "isms", "ism", "ish", "hood", "ship", "wards", "ward", "like", "dom",
                                  "ton", "son", "ham", "ley", "worth", "wood", "field", "ford", "land", "man", "burg",
                                  "stein", "berg", "shire", "ville", "by"]
    static let secondarySuffixes = ["ization", "izations", "ations", "ation", "ators", "ator", "ates", "ated", "ating",
                                    "ate", "izes", "ized", "izing", "izer", "ize", "ises", "ised", "ising", "ise",
                                    "ifies", "ified", "ify"]

    /// Returns (primary nucleus index, secondary nucleus indices).
    func assignStress(_ phones: [Phone], word: String) -> (Int, [Int])? {
        let allNuclei = phones.indices.filter { LetterToSound.vowelPhones.contains(phones[$0].p) }
        guard !allNuclei.isEmpty else { return nil }
        if allNuclei.count == 1 { return (allNuclei[0], []) }
        var secondary: [Int] = []
        // Strip stress-neutral suffixes (keep at least one vowel group in the stem).
        var stemEnd = word.count
        var stem = word
        var changed = true
        while changed {
            changed = false
            for suf in LetterToSound.neutralSuffixes where stem.hasSuffix(suf) && stem.count - suf.count >= 3 {
                let cand = String(stem.dropLast(suf.count))
                let nucleiInCand = allNuclei.filter { phones[$0].letter < cand.count }
                guard nucleiInCand.count >= 1, cand.contains(where: { "aeiouy".contains($0) }) else { continue }
                if suf == "s" && (cand.hasSuffix("s") || cand.hasSuffix("u") || cand.hasSuffix("i")) { continue }
                stem = cand
                stemEnd = cand.count
                changed = true
                break
            }
        }
        var nuclei = allNuclei.filter { phones[$0].letter < stemEnd }
        if nuclei.isEmpty { nuclei = allNuclei }
        // "un-" keeps a secondary stress.
        var first = 0
        if word.hasPrefix("un"), word.count > 4, !"aeiou".contains(Array(word)[2]), nuclei.count >= 2 {
            secondary.append(nuclei[0])
            first = 1
        }
        let n = stem.count
        func nucleusBefore(letter: Int) -> Int? { nuclei.last(where: { phones[$0].letter < letter }) }
        func nucleusAt(letter: Int) -> Int? { nuclei.first(where: { phones[$0].letter >= letter }) }
        var primary: Int? = nil
        let stemE = stem != word && !stem.hasSuffix("e") ? stem + "e" : stem
        for suf in LetterToSound.secondarySuffixes where (stem.hasSuffix(suf) || stemE.hasSuffix(suf)) && n > suf.count + 1 {
            let at = stem.hasSuffix(suf) ? n - suf.count : n + 1 - suf.count
            if let k = nucleusAt(letter: at), k != nuclei[first] {
                if suf.hasPrefix("ization") || suf.hasPrefix("ation") {
                    primary = k == nuclei.last ? k : nuclei.first(where: { $0 > k && phones[$0].p == "EY" }) ?? k
                    if suf.hasPrefix("ization") { secondary.append(k) }
                } else {
                    secondary.append(k)
                }
            }
            break
        }
        let rest = Array(nuclei[first...]).filter { !secondary.contains($0) || $0 == primary }
        if primary == nil {
            for suf in LetterToSound.stressedSuffixes where stem.hasSuffix(suf) && n > suf.count + 2 {
                if let k = nucleusAt(letter: n - suf.count) { primary = k; break }
            }
        }
        if primary == nil {
            for suf in ["ology", "ography", "onomy", "ometer", "ometry", "osophy", "ologist", "ographer"] where stem.hasSuffix(suf) {
                if let k = nucleusAt(letter: n - suf.count) { primary = k; break }
            }
        }
        if primary == nil {
            for suf in LetterToSound.preStressSuffixes where stem.hasSuffix(suf) && n > suf.count + 1 {
                if let k = nucleusBefore(letter: n - suf.count) { primary = k; break }
            }
        }
        if primary == nil {
            let cands = rest.isEmpty ? nuclei : rest
            if cands.count == 1 {
                primary = cands[0]
            } else if cands.count == 2 {
                primary = cands[0]
                for pre in LetterToSound.unstressedPrefixes where stem.hasPrefix(pre) && stem.count >= pre.count + 3 {
                    if phones[cands[0]].letter < pre.count && phones[cands[1]].letter >= pre.count { primary = cands[1]; break }
                }
            } else {
                func heavy(_ k: Int) -> Bool {
                    let idx = cands[k]
                    if LetterToSound.longPhones.contains(phones[idx].p) { return true }
                    let next = k + 1 < cands.count ? cands[k + 1] : phones.count
                    return next - idx - 1 >= 2
                }
                let penult = cands.count - 2
                if "aoi".contains(stem.last!) && (stem == word || stem + "s" == word) && phones[cands[penult]].p != "AX" {
                    primary = cands[penult]  // Italian/Spanish/Japanese-looking names
                } else if heavy(penult) && phones[cands[penult]].p != "AX" {
                    primary = cands[penult]
                } else {
                    primary = cands[cands.count - 3]
                }
            }
        }
        guard let p = primary else { return nil }
        // Secondary stress two syllables before the primary (alternating rhythm).
        if let pi = allNuclei.firstIndex(of: p), pi >= 2 {
            let k = allNuclei[pi - 2]
            if !secondary.contains(k) && (first == 0 || k != allNuclei[0]) && phones[k].p != "AX" { secondary.append(k) }
        }
        return (p, secondary.filter { $0 != p })
    }

    func rules(_ word: String) -> String? {
        var phones = applyRules(word)
        // Final -s / -ed voicing by the preceding sound.
        let voiceless: Set<String> = ["p", "t", "k", "f", "TH", "SH", "CH", "s"]
        if word.count > 3, word.hasSuffix("s"), !word.hasSuffix("ss"), !word.hasSuffix("us"), !word.hasSuffix("is"),
           let last = phones.last, last.p == "s" || last.p == "z", phones.count >= 2 {
            phones[phones.count - 1].p = voiceless.contains(phones[phones.count - 2].p) ? "s" : "z"
        }
        if word.count > 4, word.hasSuffix("ed"), !word.hasSuffix("eed"), let last = phones.last, last.p == "d" || last.p == "t",
           phones.count >= 2 {
            let prev = phones[phones.count - 2].p
            if prev == "t" || prev == "d" {
                phones.insert(Phone(p: "IX", letter: word.count - 2), at: phones.count - 1)
                phones[phones.count - 1].p = "d"
            } else if LetterToSound.vowelPhones.contains(prev) && prev != "IH" {
                phones[phones.count - 1].p = "d"
            } else {
                phones[phones.count - 1].p = voiceless.contains(prev) ? "t" : "d"
            }
        }
        guard let (primary, secondary) = assignStress(phones, word: word) else {
            return render(phones, primary: nil, secondary: [])
        }
        // Reduce unstressed vowels.
        for (i, ph) in phones.enumerated() where i != primary && !secondary.contains(i) {
            switch ph.p {
            case "EH" where british: phones[i].p = "IH"
            case "AE", "AH", "EH", "AA", "OL", "AO": phones[i].p = "AX"
            case "ER": phones[i].p = "AXR"
            case "IH" where !british:
                // American: medial unstressed /ɪ/ is mostly schwa, but not in -ing, -ic, -ish, -ige/-age endings.
                let next = i + 1 < phones.count ? phones[i + 1].p : ""
                let isLastVowel = !phones[(i + 1)...].contains(where: { LetterToSound.vowelPhones.contains($0.p) })
                let keep = next == "NG" || (isLastVowel && ["k", "SH", "JH", "CH", "g", "s", "z", "t", "d", "v", "f", "p", "b", "m", "TH"].contains(next)) || i == 0
                if !keep { phones[i].p = "AX" }
            default: break
            }
        }
        return render(phones, primary: primary, secondary: secondary, word: word)
    }

    func render(_ phones: [Phone], primary: Int?, secondary: [Int], word: String = "") -> String {
        let us: [String: String] = ["AA": "ɑ", "OL": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "W", "AX": "ə",
                                    "AXR": "əɹ", "AY": "I", "EH": "ɛ", "ER": "ɜɹ", "EY": "A", "IH": "ɪ", "IX": "ᵻ", "IY": "i",
                                    "OW": "O", "OY": "Y", "UH": "ʊ", "UW": "u",
                                    "b": "b", "CH": "ʧ", "d": "d", "DH": "ð", "f": "f", "g": "ɡ", "h": "h", "JH": "ʤ",
                                    "k": "k", "l": "l", "m": "m", "n": "n", "NG": "ŋ", "p": "p", "r": "ɹ", "s": "s",
                                    "SH": "ʃ", "t": "t", "TH": "θ", "v": "v", "w": "w", "y": "j", "z": "z", "ZH": "ʒ"]
        var gb = us
        gb["AA"] = "ɑː"; gb["OL"] = "ɒ"; gb["AE"] = "a"; gb["AO"] = "ɔː"; gb["AXR"] = "ə"; gb["ER"] = "ɜː"
        gb["OW"] = "Q"; gb["IY"] = "i"; gb["UW"] = "u"; gb["IX"] = "ɪ"
        let table = british ? gb : us
        var out = ""
        for (i, ph) in phones.enumerated() {
            let isVowel = LetterToSound.vowelPhones.contains(ph.p)
            if isVowel && i == primary { out += "ˈ" } else if isVowel && secondary.contains(i) { out += "ˌ" }
            var sym = table[ph.p] ?? ""
            if british {
                let isLast = i == phones.count - 1 || (i == phones.count - 2 && ["z", "d"].contains(phones[i + 1].p))
                if ph.p == "UW" || (ph.p == "IY" && (i == primary || !isLast)) { sym += "ː" }
                // /juː/ after coronals: "tune", "dubious", "new" (from the letter u).
                if ph.p == "UW", i > 0, ["t", "d", "n", "s", "z", "l"].contains(phones[i - 1].p) {
                    let w = Array(word.uppercased())
                    if ph.letter < w.count, w[ph.letter] == "U", !(ph.letter > 0 && "OR".contains(w[ph.letter - 1])), !(phones[i - 1].p == "l" && i == 1) {
                        out += "j"
                    }
                }
                // Unstressed -ion/-ian after a fricative: syllabic n.
                if ph.p == "AX", i != primary, i + 2 == phones.count, phones[i + 1].p == "n", i > 0, ["SH", "ZH"].contains(phones[i - 1].p) { sym = "ᵊ" }
                if ph.p == "r" {
                    // Non-rhotic: r only before a vowel; otherwise lengthen/centre the vowel.
                    let nextIsVowel = i + 1 < phones.count && LetterToSound.vowelPhones.contains(phones[i + 1].p)
                    if !nextIsVowel {
                        let prev = i > 0 ? phones[i - 1].p : ""
                        switch prev {
                        case "EH": out += "ː"
                        case "IH", "UH", "AY", "AW": out += "ə"
                        case "OW", "AH":
                            if out.hasSuffix("Q") { out.removeLast(); out += "ɔː" }
                        default: break
                        }
                        sym = ""
                    }
                }
            } else if ph.p == "t", i > 0, i + 1 < phones.count {
                // American flap: vowel/r + t + unstressed vowel.
                let prev = phones[i - 1].p, next = phones[i + 1].p
                if (LetterToSound.vowelPhones.contains(prev) || prev == "r") && LetterToSound.vowelPhones.contains(next) && i + 1 != primary {
                    sym = "ɾ"
                }
            }
            // Syllabic l after a consonant at the end: "-ble", "-tle" → ᵊl.
            if ph.p == "AX", i + 1 < phones.count, phones[i + 1].p == "l", i + 2 == phones.count || (i + 3 == phones.count && ["IY", "z"].contains(phones[i + 2].p)), i > 0, !LetterToSound.vowelPhones.contains(phones[i - 1].p), i != primary {
                sym = "ᵊ"
            }
            out += sym
        }
        if primary == nil, !out.isEmpty {
            return Phon.applyStress(out, 2) ?? out
        }
        return out
    }
}
