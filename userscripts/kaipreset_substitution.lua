-- Word substitution
-- Performs a search-and-replace on the AI's output.

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


local example_config = [[;-- Substitution
;--
;-- This example config causes all occurrences of "Hello," (without the double
;-- quotes) to be replaced with "Goodbye," (without the double quotes) and
;-- all occurrences of "test" to be replaced with "****".
;--
;-- The strings are parsed as Lua strings, so the standard escape sequences \",
;-- \n, \\, and so on apply here as well.
;--
return {
    {"Hello,", "Goodbye,"},
    {"test", "****"},
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


function userscript.outmod()
    for i, output in ipairs(kobold.outputs) do
        for j, row in ipairs(cfg) do
            output = output:gsub(row[1], row[2])
        end
        kobold.outputs[i] = output
    end
end

return userscript
