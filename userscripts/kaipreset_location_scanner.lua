-- Location scanner
-- Activates world info entries based on what the AI thinks the current location
-- is.

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


local example_config = [[;-- Location scanner
;--
;-- Usage instructions:
;--
;-- 1. Create a world info folder with name containing the string
;--    "<||ls||>" (without the double quotes).  The name can be anything as
;--    long as it contains that inside it somewhere -- for example, you could
;--    set the name to "Locations <||ls||>".
;--
;-- 2. Create a non-selective, constant world info key _in that folder_ with key
;--    "<||lslocation||>" (without the double quotes).  Every once in a while,
;--    this script will generate 20 tokens using "The current location"
;--    as the submission and save the output into the <||lslocation||> entry.
;--
;-- 3. Put some other world info entries into the world info folder.  These
;--    entries will _only_ be triggered by the contents of the <||lslocation||>
;--    entry and not by your story itself, or if it has constant key turned on.
;--
;-- You can edit some of the configuration values below to modify some of this
;-- behaviour:
;--
return {
    location_folder = "<||ls||>",
    location_key = "<||lslocation||>",
    submission = "\n\nThe current location:",
    n_wait = 12,  -- The script will run its extra generation every time your story grows by this many chunks.
    n_tokens = 20,  -- Number of tokens to generate in extra generation
    singleline = true,  -- true or false; true will result in the extra generation's output being cut off after the first line.
    trim = true,  -- true or false; true will result in the extra generation's output being cut off after the end of its last sentence.
    include = false,  -- true or false; true will result in the <||lslocation||> entry's content being included in the story.
    template = "<|>",  -- Allows you to format the extra generation's output; for example, to surround the output in square brackets, set this to "[<|>]"
}
]]

local cfg  ---@type table<string, any>
do
    -- If config file is empty, write example config
    local f <close> = kobold.get_config_file()
    f:seek("set")
    if f:read(1) == nil then
        f:write(example_config)
    end
    f:seek("set")
    example_config = nil

    -- Read config
    local err
    cfg, err = load(f:read("a"))
    if err ~= nil then
        error(err)
    end
    cfg = cfg()
end

if cfg.include == nil then
    cfg.include = false
elseif cfg.template == nil then
    cfg.template = "<|>"
end


local folder  ---@type KoboldWorldInfoFolder|nil
local entry  ---@type KoboldWorldInfoEntry|nil
local location = ""
local orig_entry_map = {}  ---@type table<integer, KoboldWorldInfoEntry>
local repeated = false
local last_quotient = math.huge

local genamt = 0

function userscript.inmod()
    if repeated then
        kobold.submission = cfg.submission
        genamt = kobold.settings.genamt
        kobold.settings.genamt = cfg.n_tokens
    end

    if entry == nil or folder == nil or not entry:is_valid() or not folder:is_valid() then
        folder = nil
        entry = nil
        for i, f in ipairs(kobold.worldinfo.folders) do
            if f.name:find(cfg.location_folder, 1, true) ~= nil then
                folder = f
                break
            end
        end
        if folder ~= nil then
            for i, e in ipairs(folder) do
                if e.key:find(cfg.location_key, 1, true) ~= nil then
                    entry = e
                    break
                end
            end
        end
    end

    orig_entry_map = {}

    if entry ~= nil then
        location = entry.content
        entry.constant = not not cfg.include
    end

    if folder ~= nil then
        for i, e in ipairs(folder) do
            if entry == nil or e.uid ~= entry.uid then
                orig_entry_map[e.uid] = {
                    constant = e.constant,
                    key = e.key,
                    keysecondary = e.keysecondary,
                }
                e.constant = e.constant or (not repeated and e:compute_context("", {scan_story=false}) ~= e:compute_context(location, {scan_story=false}))
                e.key = ""
                e.keysecondary = ""
            end
        end
    end
end

function userscript.outmod()
    if entry ~= nil and entry:is_valid() then
        entry.constant = true
    end

    if repeated then
        local output = kobold.outputs[1]
        kobold.outputs[1] = ""

        for chunk in kobold.story:reverse_iter() do
            if chunk.content ~= "" then
                chunk.content = ""
                break
            end
        end

        kobold.settings.genamt = genamt

        output = output:match("^%s*(.*)%s*$")

        print("Extra generation result (prior to formatting): " .. output)

        if cfg.singleline then
            output = output:match("^[^\n]*")
        end

        if cfg.trim then
            local i = 0
            while true do
                local j = output:find("[.?!)]", i + 1)
                if j == nil then
                    break
                end
                i = j
            end
            if i > 0 then
                if output:sub(i+1, i+1) == '"' then
                    i = i + 1
                end
                output = output:sub(1, i)
            end
        end

        location = cfg.template:gsub("<|>", output)

        print("Extra generation result (after formatting): " .. location)

        if entry ~= nil and entry:is_valid() then
            entry.content = location
        end
    end

    local size = 0
    for _ in kobold.story:forward_iter() do
        size = size + 1
    end

    for uid, orig in pairs(orig_entry_map) do
        for k, v in pairs(orig) do
            kobold.worldinfo:finduid(uid)[k] = v
        end
    end

    local quotient = math.floor(size / cfg.n_wait)
    if repeated then
        repeated = false
    elseif quotient > last_quotient then
        print("Running extra generation")
        kobold.restart_generation()
        repeated = true
    end
    last_quotient = quotient
end

return userscript
