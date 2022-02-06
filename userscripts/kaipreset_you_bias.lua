-- You bias
-- Makes the word "You" less (or more) common in character references
-- , optionally also between double quotes.
-- Only works with models with a tokenizer based on GPT-2, such as GPT-2,
-- GPT-Neo and GPT-J.

-- This file is part of KoboldAI.
--
-- KoboldAI is free software: you can redistribute it and/or modify
-- it under the terms of the GNU Affero General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU Affero General Public License for more details.
--
-- You should have received a copy of the GNU Affero General Public License
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.

kobold = require("bridge")()  -- This line is optional and is only for EmmyLua type annotations
local userscript = {}  ---@class KoboldUserScript


local example_config = [[;-- You bias
;--
return {
    bias = -7.0,  -- Negative numbers make it less likely, positive numbers more, and -math.huge impossible
    only_if_outside_double_quotes = true,
}
]]

-- If config file is empty, write example config
local f = kobold.get_config_file()
f:seek("set")
if f:read(1) == nil then
    f:write(example_config)
end
f:seek("set")
example_config = nil

-- Read config
local cfg, err = load(f:read("a"))
f:close()
if err ~= nil then
    error(err)
end
cfg = cfg()
if type(cfg.bias) ~= "number" then
    error("`bias` must be a number")
elseif cfg.bias ~= cfg.bias or cfg.bias == math.huge then
    error("`bias` can't be `nan` or `math.huge`")
end


---@type table<integer, integer>
local you_tokens <const> = {345, 921, 1639, 5832, 7013, 36981}

local genmod_run = false

function userscript.genmod()
    genmod_run = true
    local context
    if cfg.only_if_outside_double_quotes then
        context = " " .. kobold.worldinfo:compute_context(kobold.submission, {})
    end

    for i, generated_row in ipairs(kobold.generated) do
        local should_bias = true

        if cfg.only_if_outside_double_quotes then
            local str = context .. kobold.decode(generated_row)
            local last_open_quote = 0
            local last_close_quote = 0
            local i = 0
            local j = 0
            while true do
                i, j = str:find('"', j+1)
                if i == nil then
                    break
                end
                if str:sub(i-1, i-1) == " " or str:sub(i-1, i-1) == "\n" then
                    last_open_quote = j
                else
                    last_close_quote = j
                end
            end
            if last_open_quote > last_close_quote then
                should_bias = false
            end
        end

        if should_bias then
            for k, v in ipairs(you_tokens) do
                kobold.logits[i][v+1] = kobold.logits[i][v+1] + cfg.bias
            end
        end
    end
end

function userscript.outmod()
    if not genmod_run then
        warn("WARNING:  Generation modifier was not executed, so this script has had no effect")
    end
end

return userscript
