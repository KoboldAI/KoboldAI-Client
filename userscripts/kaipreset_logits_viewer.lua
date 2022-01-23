-- Logit viewer
-- Displays raw token scores and softmax probabilities during generation.

kobold = require("bridge")()
local userscript = {}  ---@class KoboldUserScript

local K = 10

---@class Pair
---@field id integer
---@field score number

---@class ArrayBase
---@type table<any, Pair>
local _ = {}

---@class Array : ArrayBase
---@field n integer

---@param array Array
---@param index integer
---@return nil
local function bubble(array, index)
    local j = 0
    while (index<<1)+1 < array.n do
        j = index
        if array[(index<<1)+1].score > array[j].score then
            j = (index<<1)+1
        end
        if (index<<1)+2 < array.n and array[(index<<1)+2].score > array[j].score then
            j = (index<<1)+2
        end
        if index == j then
            break
        end
        local b = array[index]
        array[index] = array[j]
        array[j] = b
        index = j
    end
end

---@param array Array
---@return nil
local function build(array)
    for i = (array.n-1)>>1, 0, -1 do
        bubble(array, i)
    end
end

---@param array Array
---@return Pair
local function pop(array)
    local r = array[0]
    array.n = array.n - 1
    array[0] = array[array.n]
    bubble(array, 0)
    return r
end

function userscript.genmod()
    if K > kobold.logits_cols then
        error("K must be at most the vocabulary size of the model")
    end

    if kobold.generated_cols > 0 then
        for s, logits in ipairs(kobold.logits) do
            local token = kobold.generated[s][kobold.generated_cols]
            print("Previous result for sequence " .. s .. ": [" .. kobold.decode(token):gsub("\n", "\\n") .. "] (" .. math.tointeger(token) .. ")")
        end
    end

    for s, logits in ipairs(kobold.logits) do
        local a = {}  ---@type Array
        local sum = 0.0
        for i = 0, kobold.logits_cols-1 do
            a[i] = {id = i, score = logits[i + 1]}
            a.n = i + 1
            sum = sum + math.exp(logits[i + 1])
        end
        build(a)
        print()
        print("Top " .. K .. " scores for sequence " .. s .. ":")
        for i = 1, K do
            local e = pop(a)
            print(("%.6f"):format(e.score), ("%.3f%%   "):format(100 * (math.exp(e.score) / sum)), e.id, "[" .. (kobold.decode(e.id):gsub("\n", "\\n")) .. "]")
        end
    end
end

return userscript
