import Foundation
import Testing
@testable import Agent

/// Tool-call arguments as models really write them.
struct ToolArgsTests {
    @Test func yesNoArgumentsTakeEverySpelling() throws {
        // Edge0's engine sent "background": "True" (its tool-call format writes parameters as text).
        for yes in [#"true"#, #""true""#, #""True""#, #""TRUE""#, #""1""#, #""yes""#, #"1"#] {
            #expect(try ToolArgs(json: #"{"background": \#(yes)}"#).bool("background"), "\(yes)")
        }
        for no in [#"false"#, #""false""#, #""False""#, #""0""#, #""no""#, #"0"#, #""maybe""#] {
            #expect(try !ToolArgs(json: #"{"background": \#(no)}"#).bool("background"), "\(no)")
        }
        #expect(try !ToolArgs(json: "{}").bool("background"))
        #expect(ToolArgs.flag(nil) == nil && ToolArgs.flag("maybe") == nil && ToolArgs.flag(" False ") == false)
    }

    @Test func numbersWrittenAsTextAreRead() throws {
        #expect(try ToolArgs(json: #"{"timeout": "300"}"#).int("timeout") == 300)
        #expect(try ToolArgs(json: #"{"timeout": 300.0}"#).int("timeout") == 300)
    }
}
