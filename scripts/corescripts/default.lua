-- Default core script
-- Runs all input modifiers and generation modifiers in forward order, and
-- runs all output modifiers in reverse order

kobold, koboldcore = require("bridge")()  -- This line is optional and is only for EmmyLua type annotations

---@class KoboldCoreScript
local corescript = {}


-- Run all the input modifiers from top to bottom
function corescript.inmod()
    for i, userscript in ipairs(koboldcore.userscripts) do
        if userscript.inmod ~= nil then
            userscript.inmod()
        end
    end
end

-- Run all the generation modifiers from top to bottom
function corescript.genmod()
    for i, userscript in ipairs(koboldcore.userscripts) do
        if userscript.genmod ~= nil then
            userscript.genmod()
        end
    end
end

-- Run all the generation modifiers from bottom to top
function corescript.outmod()
    local userscript
    for i = #koboldcore.userscripts, 1, -1 do
        userscript = koboldcore.userscripts[i]
        if userscript.outmod ~= nil then
            userscript.outmod()
        end
    end
end


return corescript
