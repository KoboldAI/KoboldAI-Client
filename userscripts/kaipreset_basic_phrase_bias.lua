-- Basic phrase bias
-- Makes certain sequences of tokens more or less likely to appear than normal.

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


---@class PhraseBiasEntry
---@field starting_bias number
---@field ending_bias number
---@field tokens table<integer, integer>
---@field n_tokens integer

local example_config = [[# Phrase bias
#
# For each phrase you want to bias, add a new line into
# this config file as a comma-separated list in this format:
# <starting bias>, <ending bias>, <comma-separated list of token IDs>
# For <starting bias> and <ending bias>, this script accepts floating point
# numbers or -inf, where positive bias values make it more likely for tokens
# to appear, negative bias values make it less likely and -inf makes it
# impossible.
#
# Example 1 (makes it impossible for the word "CHAPTER", case-sensitive, to
# appear at the beginning of a line in the output):
# -inf, -inf, 41481
#
# Example 2 (makes it unlikely for the word " CHAPTER", case-sensitive, with
# a leading space, to appear in the output, with the unlikeliness increasing
# even more if the first token " CH" has appeared):
# -10.0, -20.0, 5870, 29485
#
# Example 3 (makes it more likely for " let the voice of love take you higher",
# case-sensitive, with a leading space, to appear in the output, with the
# bias increasing as each consecutive token in that phrase appears):
# 7, 25.4, 1309, 262, 3809, 286, 1842, 1011, 345, 2440
#
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
print("Loading phrase bias config...")
local bias_array = {}  ---@type table<integer, PhraseBiasEntry>
local bias_array_count = 0
local val_count = 0
local line_count = 0
local row = {}  ---@type PhraseBiasEntry
local val_orig
for line in f:lines("l") do
    line_count = line_count + 1
    if line:find("^ *#") == nil and line:find("%S") ~= nil then
        bias_array_count = bias_array_count + 1
        val_count = 0
        row = {}
        row.tokens = {}
        row.n_tokens = 0
        for val in line:gmatch("[^,%s]+") do
            val_count = val_count + 1
            val_orig = val
            if val_count <= 2 then
                val = val:lower()
                if val:sub(-3) == "inf" then
                    val = math.tointeger(val:sub(1, -4) .. "1")
                    if val ~= val or type(val) ~= "number" or val > 0 then
                        f:close()
                        error("First two values of line " .. line_count .. " of config file must be finite floating-point numbers or -inf, but got '" .. val_orig .. "' as value #" .. val_count)
                    end
                    val = val * math.huge
                else
                    val = tonumber(val)
                    if val ~= val or type(val) ~= "number" then
                        f:close()
                        error("First two values of line " .. line_count .. " of config file must be finite floating-point numbers or -inf, but got '" .. val_orig .. "' as value #" .. val_count)
                    end
                end
                if val_count == 1 then
                    row.starting_bias = val
                else
                    row.ending_bias = val
                end
            else
                val = math.tointeger(val)
                if type(val) ~= "number" or val < 0 then
                    f:close()
                    error("All values after the first two values of line " .. line_count .. " of config file must be nonnegative integers, but got '" .. val_orig .. "' as value #" .. val_count)
                end
                row.n_tokens = row.n_tokens + 1
                row.tokens[row.n_tokens] = val
            end
        end
        if val_count < 3 then
            f:close()
            error("Line " .. line_count .. " of config file must contain at least 3 values, but found " .. val_count)
        end
        bias_array[bias_array_count] = row
    end
end
f:close()
print("Successfully loaded " .. bias_array_count .. " phrase bias entr" .. (bias_array_count == 1 and "y" or "ies") .. ".")


local genmod_run = false

---@param starting_val number
---@param ending_val number
---@param factor number
---@return number
local function logit_interpolate(starting_val, ending_val, factor)
    -- First use the logistic function on the start and end values
    starting_val = 1/(1 + math.exp(-starting_val))
    ending_val = 1/(1 + math.exp(-ending_val))

    -- Use linear interpolation between these two values
    local val = starting_val + factor*(ending_val - starting_val)

    -- Return logit of this value
    return math.log(val/(1 - val))
end


function userscript.genmod()
    genmod_run = true

    local context_tokens = kobold.encode(kobold.worldinfo:compute_context(kobold.submission))
    local factor  ---@type number
    local next_token  ---@type integer
    local sequences = {}  ---@type table<integer, table<integer, integer>>
    local n_tokens = 0
    local max_overlap = {}  ---@type table<integer, integer>

    local biased_tokens = {}  ---@type table<integer, table<integer, boolean>>
    for i = 1, kobold.generated_rows do
        biased_tokens[i] = {}
    end

    -- For each partially-generated sequence...
    for i, generated_row in ipairs(kobold.generated) do

        -- Build an array `tokens` as the concatenation of the context
        -- tokens and the generated tokens of this sequence

        local tokens = {}
        n_tokens = 0
        for k, v in ipairs(context_tokens) do
            n_tokens = n_tokens + 1
            tokens[n_tokens] = v
        end
        for k, v in ipairs(generated_row) do
            n_tokens = n_tokens + 1
            tokens[n_tokens] = v
        end

        -- For each phrase bias entry in the config file...
        for _, bias_entry in ipairs(bias_array) do

            -- Determine the largest integer `max_overlap[i]` such that the last
            -- `max_overlap[i]` elements of `tokens` equal the first
            -- `max_overlap[i]` elements of `bias_entry.tokens`

            max_overlap[i] = 0
            local s = {}
            local z = {[0] = 0}
            local l = 0
            local r = 0
            local n_s = math.min(n_tokens, bias_entry.n_tokens)
            local j = 0
            for k = 1, n_s do
                s[j] = bias_entry.tokens[k]
                j = j + 1
            end
            for k = n_tokens - n_s + 1, n_tokens do
                s[j] = tokens[k]
                j = j + 1
            end
            for k = 1, (n_s<<1) - 1 do
                if k <= r and z[k - l] - 1 < r - k then
                    z[k] = z[k - l]
                else
                    l = k
                    if k > r then
                        r = k
                    end
                    while r < (n_s<<1) and s[r - l] == s[r] do
                        r = r + 1
                    end
                    z[k] = r - l
                    r = r - 1
                end
                if z[k] <= n_s and z[k] == (n_s<<1) - k then
                    max_overlap[i] = z[k]
                    break
                end
            end
        end
    end

    -- For each phrase bias entry in the config file...
    for _, bias_entry in ipairs(bias_array) do

        -- For each partially-generated sequence...
        for i, generated_row in ipairs(kobold.generated) do

            -- Use `max_overlap` to determine which token in the bias entry to
            -- apply bias to

            if max_overlap[i] == 0 or max_overlap[i] == bias_entry.n_tokens then
                if bias_entry.tokens[2] == nil then
                    factor = 1
                else
                    factor = 0
                end
                next_token = bias_entry.tokens[1]
            else
                factor = max_overlap[i]/(bias_entry.n_tokens - 1)
                next_token = bias_entry.tokens[max_overlap[i]+1]
            end

            -- Apply bias

            if not biased_tokens[i][next_token] then
                kobold.logits[i][next_token + 1] = kobold.logits[i][next_token + 1] + logit_interpolate(bias_entry.starting_bias, bias_entry.ending_bias, factor)
                biased_tokens[i][next_token] = true
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
