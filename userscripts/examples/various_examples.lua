-- Example script
-- Description goes on
--[[subsequent lines including
in multiline comments]]

kobold = require("bridge")()  -- This line is optional and is only for EmmyLua type annotations

-- You can import libraries that are in extern/lualibs/
local inspect = require("inspect")
local mt19937ar = require("mt19937ar")


---@class KoboldUserScript
local userscript = {}


local twister = mt19937ar.new()
local seed = math.random(0, 2147483647)

local token_num = 0
local lifetime_token_num = 0

-- This gets run when user submits a string to the AI (right after the input
-- formatting is applied but before the string is actually sent to the AI)
function userscript.inmod()
    warn("\nINPUT MODIFIER")
    token_num = 0
    twister:init_genrand(seed)
    print("Submitted text: " .. kobold.submission)  -- You can also write to kobold.submission to alter the user's input
    print("top-p sampling value: " .. kobold.settings.settopp)
end

-- This gets run every time the AI generates a token (before the token is
-- actually sampled, so this is where you can make certain tokens more likely
-- to appear than others)
function userscript.genmod()
    warn("\nGENERATION MODIFIER")

    print("Tokens generated in the current generation: " .. token_num)
    print("Tokens generated since this script started up: " .. lifetime_token_num)

    local r = twister:genrand_real3()
    print("Setting top-p sampling value to " .. r)
    kobold.settings.settopp = r

    local generated = {}
    for sequence_number, tokens in ipairs(kobold.generated) do
        generated[sequence_number] = kobold.decode(tokens)
    end
    print("Current generated strings: " .. inspect(generated))

    if token_num == math.floor(kobold.settings.genamt/2) then
        print("\n\n\n\n\n\nMaking all subsequent tokens more likely to be exclamation marks...")
    end
    if token_num >= math.floor(kobold.settings.genamt/2) then
        for i = 1, kobold.settings.numseqs do
            kobold.logits[i][1] = 13.37
        end
    end

    token_num = token_num + 1
    lifetime_token_num = lifetime_token_num + 1
end

-- This gets run right before the output formatting is applied after generation
-- is finished
function userscript.outmod()
    warn("\nOUTPUT MODIFIER")
    for chunk in kobold.story:reverse_iter() do
        print(chunk.num, chunk.content)
    end
    print("Wrapping first output in brackets")
    kobold.outputs[1] = "[" .. kobold.outputs[1] .. "]"
end

return userscript
