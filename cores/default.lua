-- Default core script
-- Runs all generation modifiers and output modifiers in forward order, and
-- runs all input modifiers in reverse order

kobold, koboldcore = require("bridge")()  -- This line is optional and is only for EmmyLua type annotations

---@class KoboldCoreScript
local corescript = {}


-- Run all the input modifiers from bottom to top
function corescript.inmod()
    for i = #koboldcore.userscripts, 1, -1 do
        local userscript = koboldcore.userscripts[i]
        userscript.inmod()
    end
end

-- Run all the generation modifiers from top to bottom
function corescript.genmod()
    for i, userscript in ipairs(koboldcore.userscripts) do
        userscript.genmod()
    end
end

-- Run all the generation modifiers from top to bottom
function corescript.outmod()
    for i, userscript in ipairs(koboldcore.userscripts) do
        userscript.outmod()
    end
end


return corescript
