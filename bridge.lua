-- KoboldAI Lua 5.4 Bridge


---@param _python? table<string, any>
---@param _bridged? table<string, any>
---@return KoboldLib, KoboldCoreLib|nil
return function(_python, _bridged)

    --==========================================================================
    -- Globally allows using a _kobold_next metamethod for "Kobold" classes only
    --==========================================================================

    local old_next = next
    ---@generic K, V
    ---@param t table<K, V>
    ---@param k? K
    ---@return K|nil, V|nil
    function next(t, k)
        local meta = getmetatable(t)
        return ((meta ~= nil and string.match(rawget(t, "_name"), "^Kobold") and type(meta._kobold_next) == "function") and meta._kobold_next or old_next)(t,k)
    end


    --==========================================================================
    -- General utility functions
    --==========================================================================

    ---@generic T
    ---@param original T
    ---@return T
    local function deepcopy(original)
        if type(original) == "table" then
            local copy = {}
            for k, v in old_next, original, nil do
                copy[k] = deepcopy(v)
            end
            setmetatable(copy, deepcopy(getmetatable(original)))
            return copy
        end
        return original
    end

    ---@param path string
    ---@return nil
    function set_require_path(path)
        local config = {}
        local i = 1
        for substring in string.gmatch(package.config, "[^\n]+") do
            config[i] = substring
            i = i + 1
        end
        package.path = path .. config[1] .. config[3] .. ".lua" .. config[2] .. path .. config[1] .. config[3] .. config[1] .. "init.lua"
        package.cpath = ""
    end


    --==========================================================================
    -- _bridged preprocessing
    --==========================================================================

    local bridged = {}
    for k, v in pairs(_bridged) do
        bridged[k] = type(v) == "userdata" and _python.as_attrgetter(v) or v
    end
    set_require_path(bridged.lib_path)


    --==========================================================================
    -- Wraps most functions in this file so that they restore original
    -- metatables prior to executing the function body
    --==========================================================================

    ---@class Metatables
    local metatables = {}
    local type_map = {
        _nil = nil,
        _boolean = false,
        _number = 0,
        _string = "",
        _function = type,
        _thread = coroutine.create(function() end),
    }

    function metatables:overwrite()
        for k, v in pairs(type_map) do
            self[k] = debug.getmetatable(v)
        end
    end

    function metatables:restore()
        for k, v in pairs(type_map) do
            debug.setmetatable(v, self[k])
        end
    end

    local metatables_original = deepcopy(metatables)
    metatables_original:overwrite()

    local metawrapper = {}
    ---@generic T : table
    ---@param t T
    ---@return T
    function metawrapper.__newindex(t, k, v)
        if type(v) == "function" then
            return rawset(t, k, function(...)
                metatables:overwrite()
                metatables_original:restore()
                local r = {v(...)}
                metatables:restore()
                return table.unpack(r)
            end)
        else
            return rawset(t, k, v)
        end
    end


    --==========================================================================
    -- Modules
    --==========================================================================

    ---@class KoboldLib
    ---@field memory string
    local kobold = setmetatable({}, metawrapper)
    local KoboldLib_mt = setmetatable({}, metawrapper)
    local KoboldLib_getters = setmetatable({}, metawrapper)
    local KoboldLib_setters = setmetatable({}, metawrapper)

    ---@param t KoboldLib
    function KoboldLib_mt.__index(t, k)
        local getter = KoboldLib_getters[k]
        if getter ~= nil then
            return getter(t)
        end
        return rawget(t, k)
    end

    ---@param t KoboldLib
    function KoboldLib_mt.__newindex(t, k, v)
        local setter = KoboldLib_setters[k]
        if setter ~= nil then
            return setter(t, v)
        end
        return rawset(t, k, v)
    end

    ---@class KoboldCoreLib
    ---@field userscripts KoboldUserScriptList
    local koboldcore = setmetatable({}, metawrapper)
    local KoboldCoreLib_mt = setmetatable({}, metawrapper)
    local KoboldCoreLib_getters = setmetatable({}, metawrapper)
    local KoboldCoreLib_setters = setmetatable({}, metawrapper)

    ---@param t KoboldCoreLib
    function KoboldCoreLib_mt.__index(t, k)
        local getter = KoboldCoreLib_getters[k]
        if getter ~= nil then
            return getter(t, k)
        end
        return rawget(t, k)
    end

    ---@param t KoboldCoreLib
    function KoboldCoreLib_mt.__newindex(t, k, v)
        local setter = KoboldCoreLib_setters[k]
        if setter ~= nil then
            return setter(t, k, v)
        end
        return rawset(t, k, v)
    end

    ---@class KoboldBridgeLib
    local koboldbridge = setmetatable({}, metawrapper)


    --==========================================================================
    -- Userscript API: World Info
    --==========================================================================

    local inmods = setmetatable({}, metawrapper)
    local genmods = setmetatable({}, metawrapper)
    local outmods = setmetatable({}, metawrapper)

    local genmod_comparison_context = nil
    local generating = true
    local userstate = "inmod"

    local fields = setmetatable({}, metawrapper)

    ---@param t KoboldWorldInfoEntry|KoboldWorldInfoFolder|KoboldWorldInfo|KoboldWorldInfoFolderSelector
    ---@return boolean
    local function check_validity(t)
        if not t:is_valid() then
            error("Attempted to use a nonexistent/deleted `"..rawget(t, "_name").."`")
            return false
        end
        return true
    end

    ---@return nil
    local function maybe_save_genmod_comparison_context()
        if userstate == "genmod" and genmod_comparison_context == nil then
            genmod_comparison_context = kobold.worldinfo:compute_context()
        end
    end


    ----------------------------------------------------------------------------

    ---@class KoboldWorldInfoEntry_base
    ---@type table<integer, nil>
    local _ = {}

    ---@class KoboldWorldInfoEntry : KoboldWorldInfoEntry_base
    ---@field key string
    ---@field keysecondary string
    ---@field content string
    ---@field comment string
    ---@field folder integer
    ---@field num integer
    ---@field selective boolean
    ---@field constant boolean
    ---@field uid integer
    local KoboldWorldInfoEntry = setmetatable({
        _name = "KoboldWorldInfoEntry",
    }, metawrapper)
    fields.KoboldWorldInfoEntry = {
        "key",
        "keysecondary",
        "content",
        "comment",
        "folder",
        "num",
        "selective",
        "constant",
        "uid",
    }
    local KoboldWorldInfoEntry_mt = setmetatable({}, metawrapper)

    local KoboldWorldInfoEntry_fieldtypes = {
        key = "string",
        keysecondary = "string",
        content = "string",
        comment = "string",
        selective = "boolean",
        constant = "boolean",
    }

    ---@return boolean
    function KoboldWorldInfoEntry:is_valid()
        return bridged.worldinfo_u.get(self.uid) ~= nil
    end

    ---@return string
    function KoboldWorldInfoEntry:compute_context()
        if not check_validity(self) then
            return ""
        end
        return bridged.compute_context({self.uid})
    end

    ---@generic K
    ---@param t KoboldWorldInfoEntry|KoboldWorldInfoFolder|KoboldWorldInfo|KoboldWorldInfoFolderSelector
    ---@param k K
    ---@return K, any
    function KoboldWorldInfoEntry_mt._kobold_next(t, k)
        return next(fields[rawget(t, "_name")], k)
    end

    ---@param t KoboldWorldInfoEntry|KoboldWorldInfoFolder|KoboldWorldInfo|KoboldWorldInfoFolderSelector
    ---@return function, KoboldWorldInfoEntry|KoboldWorldInfoFolder|KoboldWorldInfo|KoboldWorldInfoFolderSelector, nil
    function KoboldWorldInfoEntry_mt.__pairs(t)
        return next, t, nil
    end

    ---@param t KoboldWorldInfoEntry
    function KoboldWorldInfoEntry_mt.__index(t, k)
        if not check_validity(t) then
            return
        elseif k == "uid" then
            return rawget(t, "_uid")
        elseif type(k) == "string" then
            return bridged.get_attr(t.uid, k)
        end
    end

    ---@param t KoboldWorldInfoEntry
    ---@return KoboldWorldInfoEntry
    function KoboldWorldInfoEntry_mt.__newindex(t, k, v)
        if not check_validity(t) then
            return
        elseif fields[rawget(t, "_name")] then
            if type(k) == "string" and KoboldWorldInfoEntry_fieldtypes[k] == nil then
                error("`"..rawget(t, "_name").."."..k.."` is a read-only attribute")
                return
            elseif type(k) == "string" and type(v) ~= KoboldWorldInfoEntry_fieldtypes[k] then
                error("`"..rawget(t, "_name").."."..k.."` must be a "..KoboldWorldInfoEntry_fieldtypes[k].."; you attempted to set it to a "..type(v))
                return
            else
                if k ~= "comment" then
                    maybe_save_genmod_comparison_context()
                end
                bridged.set_attr(t.uid, k, v)
                return t
            end
        end
        return rawset(t, k, v)
    end


    ----------------------------------------------------------------------------

    ---@class KoboldWorldInfoFolder_base
    ---@type table<integer, KoboldWorldInfoEntry>
    local _ = {}

    ---@class KoboldWorldInfoFolder : KoboldWorldInfoFolder_base
    ---@field uid integer
    ---@field comment string
    local KoboldWorldInfoFolder = setmetatable({
        _name = "KoboldWorldInfoFolder",
    }, metawrapper)

    fields.KoboldWorldInfoFolder = {
        "uid",
    }
    local KoboldWorldInfoFolder_mt = setmetatable({}, metawrapper)

    ---@param u integer
    ---@return KoboldWorldInfoEntry|nil
    function KoboldWorldInfoFolder:finduid(u)
        if not check_validity(self) or type(u) ~= "number" then
            return
        end
        local query = bridged.worldinfo_u.get(u)
        if query == nil or (rawget(self, "_name") == "KoboldWorldInfoFolder" and self.uid ~= query.get("folder")) then
            return
        end
        local entry = deepcopy(KoboldWorldInfoEntry)
        rawset(entry, "_uid", u)
        return entry
    end

    ---@param entries? KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>
    ---@return string
    function KoboldWorldInfoFolder:compute_context(entries)
        if not check_validity(self) then
            return
        end
        if entries ~= nil and type(entries) ~= "table" or (entries.name ~= nil and entries.name ~= "KoboldWorldInfoEntry") then
            error("`compute_context` takes a KoboldWorldInfoEntry, table of KoboldWorldInfoEntries or nil as argument, but got a " .. type(entries))
            return ""
        end
        if entries.name == "KoboldWorldInfoEntry" then
            entries = {entries}
        end
        local _entries
        for k, v in pairs(entries) do
            if type(v) == "table" and v.name == "KoboldWorldInfoEntry" and (rawget(self, "_name") ~= "KoboldWorldInfoFolder" or self.uid == v.uid) and v:is_valid() then
                _entries[k] = v.uid
            end
        end
        return bridged.compute_context(_entries)
    end

    ---@return boolean
    function KoboldWorldInfoFolder:is_valid()
        return bridged.wifolders_d.get(self.uid) ~= nil
    end

    ---@param t KoboldWorldInfoFolder
    ---@return integer
    function KoboldWorldInfoFolder_mt.__len(t)
        if not check_validity(t) then
            return 0
        end
        return _python.builtins.len(bridged.worldinfo_u.get(t.uid))
    end

    KoboldWorldInfoFolder_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfoFolder_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    ---@param t KoboldWorldInfoFolder|KoboldWorldInfo
    ---@return KoboldWorldInfoEntry|nil
    function KoboldWorldInfoFolder_mt.__index(t, k)
        if not check_validity(t) then
            return
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "uid" then
            return rawget(t, "_uid")
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "comment" then
            return bridged.wifolders_d.get(t.uid).__getitem__("comment")
        elseif type(k) == "number" then
            local query = rawget(t, "_name") == "KoboldWorldInfoFolder" and bridged.wifolders_u.get(t.uid) or bridged.worldinfo
            k = math.tointeger(k)
            if k == nil or k < 1 or k > _python.builtins.len(query) then
                return
            end
            local entry = deepcopy(KoboldWorldInfoEntry)
            rawset(entry, "_uid", query.__getitem__(k))
            return entry
        end
    end

    ---@param t KoboldWorldInfoFolder|KoboldWorldInfo
    ---@return KoboldWorldInfoFolder|KoboldWorldInfo
    function KoboldWorldInfoFolder_mt.__newindex(t, k, v)
        if not check_validity(t) then
            return
        elseif type(t) == "number" and math.tointeger(t) ~= nil then
            error("Cannot write to integer indices of `"..rawget(t, "_name").."`")
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "uid" then
            error("`"..rawget(t, "_name").."."..k.."` is a read-only attribute")
        elseif t == "comment" then
            if type(v) ~= "string" then
                error("`"..rawget(t, "_name").."."..k.."` must be a string; you attempted to set it to a "..type(v))
                return
            end
            bridged.safe_setitem(bridged.wifolders_d.get(t.uid), "comment", v)
            return t
        else
            return rawset(t, k, v)
        end
    end


    ----------------------------------------------------------------------------

    ---@class KoboldWorldInfoFolderSelector_base
    ---@type table<integer, KoboldWorldInfoFolder>
    local _ = {}

    ---@class KoboldWorldInfoFolderSelector : KoboldWorldInfoFolderSelector_base
    local KoboldWorldInfoFolderSelector = setmetatable({
        _name = "KoboldWorldInfoFolderSelector",
    }, metawrapper)
    local KoboldWorldInfoFolderSelector_mt = setmetatable({}, metawrapper)

    ---@param u integer
    ---@return KoboldWorldInfoFolder|nil
    function KoboldWorldInfoFolderSelector:finduid(u)
        if not check_validity(self) or type(u) ~= "number" then
            return
        end
        local query = bridged.wifolders_d.get(u)
        if query == nil then
            return
        end
        local folder = deepcopy(KoboldWorldInfoFolder)
        rawset(folder, "_uid", u)
        return folder
    end

    ---@return boolean
    function KoboldWorldInfoFolderSelector:is_valid()
        return true
    end

    ---@param t KoboldWorldInfoFolderSelector
    ---@return integer
    function KoboldWorldInfoFolderSelector_mt.__len(t)
        if not check_validity(t) then
            return 0
        end
        return #kobold.worldinfo
    end

    KoboldWorldInfoFolderSelector_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfoFolderSelector_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    ---@param t KoboldWorldInfoFolderSelector
    ---@return KoboldWorldInfoFolder|nil
    function KoboldWorldInfoFolderSelector_mt.__index(t, k)
        if not check_validity(t) or type(t) ~= "number" or math.tointeger(t) == nil or t < 1 or t > _python.builtins.len(bridged.wifolders_l) then
            return
        end
        local folder = deepcopy(KoboldWorldInfoFolder)
        rawset(folder, "_uid", bridged.wifolders_l.__getitem__(k))
        return folder
    end

    ---@param t KoboldWorldInfoFolderSelector
    ---@return KoboldWorldInfoFolderSelector
    function KoboldWorldInfoFolderSelector_mt.__newindex(t, k, v)
        if check_validity(t) or (type(t) == "number" and math.tointeger(t) ~= nil) then
            error("Cannot write to integer indices of `"..rawget(t, "_name").."`")
        end
        return rawset(t, k, v)
    end


    ----------------------------------------------------------------------------

    ---@class KoboldWorldInfo : KoboldWorldInfoFolder_base
    local KoboldWorldInfo = setmetatable({
        _name = "KoboldWorldInfo",
    }, metawrapper)
    local KoboldWorldInfo_mt = setmetatable({}, metawrapper)

    KoboldWorldInfo.folders = KoboldWorldInfoFolderSelector

    KoboldWorldInfo.finduid = KoboldWorldInfoFolder.finduid

    KoboldWorldInfo.compute_context = KoboldWorldInfoFolder.compute_context

    ---@return boolean
    function KoboldWorldInfo:is_valid()
        return true
    end

    ---@param t KoboldWorldInfo
    ---@return integer
    function KoboldWorldInfo_mt.__len(t)
        if not check_validity(t) then
            return 0
        end
        return _python.builtins.len(bridged.worldinfo)
    end

    KoboldWorldInfo_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfo_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    KoboldWorldInfo_mt.__index = KoboldWorldInfoFolder_mt.__index

    KoboldWorldInfo_mt.__newindex = KoboldWorldInfoFolder_mt.__newindex

    kobold.worldinfo = KoboldWorldInfo


    --==========================================================================
    -- Userscript API: Settings
    --==========================================================================

    ---@class KoboldSettings_base
    ---@type table<string, any>
    local _ = {}

    ---@class KoboldSettings : KoboldSettings_base
    local KoboldSettings = setmetatable({
        _name = "KoboldSettings",
    }, metawrapper)
    local KoboldSettings_mt = setmetatable({}, metawrapper)

    ---@generic K
    ---@param t KoboldSettings
    ---@param k K
    ---@return K, any
    function KoboldSettings_mt._kobold_next(t, k)
        local v
        repeat
            k, v = next()
        until type(k) ~= "string" or k ~= "_name"
        return k, v
    end

    ---@param t KoboldSettings
    ---@return function, KoboldSettings, nil
    function KoboldSettings_mt.__pairs(t)
        return next, t, nil
    end

    ---@param t KoboldSettings
    ---@return any, boolean
    function KoboldSettings_mt.__index(t, k)
        if type(k) ~= "string" then
            return
        end
        if k == "gen_len" then
            return bridged.get_gen_len()
        elseif bridged.has_setting(k) then
            return bridged.get_setting(k), true
        else
            return nil, false
        end
    end

    ---@param t KoboldSettings_base
    function KoboldSettings_mt.__newindex(t, k, v)
        if k == "gen_len" and type(v) == "number" and math.tointeger(v) ~= nil and v >= 0 then
            bridged.set_gen_len(v)
        elseif type(k) == "string" and bridged.has_setting(k) and type(v) == type(bridged.get_setting(k)) then
            return bridged.set_setting(k, v)
        end
        return t
    end

    kobold.settings = KoboldSettings


    --==========================================================================
    -- Userscript API: Memory
    --==========================================================================

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.memory(t)
        return bridged.get_memory()
    end

    ---@param t KoboldLib
    ---@param v string
    ---@return KoboldLib
    function KoboldLib_setters.memory(t, v)
        if type(t) ~= "string" then
            error("`KoboldLib.memory` must be a string; you attempted to set it to a "..type(v))
            return
        end
        maybe_save_genmod_comparison_context()
        bridged.set_memory(v)
    end


    --==========================================================================
    -- Userscript API: Utilities
    --==========================================================================

    ---@param str string
    ---@return table<integer, integer>
    function kobold.encode(str)
        if type(str) ~= "string" then
            error("`encode` takes a string as argument, but got a " .. type(str))
            return
        end
        local encoded = {}
        local i = 1
        for token in _python.iter(bridged.encode(str)) do
            encoded[i] = token
            i = i + 1
        end
        return encoded
    end

    ---@param tok integer|table<integer, integer>
    ---@return string
    function kobold.decode(tok)
        if type(tok) ~= "number" and type(tok) ~= "table" then
            error("`decode` takes a number or table of numbers as argument, but got a " .. type(tok))
            return
        end
        if type(tok) ~= "number" then
            tok = {tok}
        end
        for k, v in ipairs(tok) do
            tok[k] = math.tointeger(v)
            if tok[k] == nil then
                error "`decode` got a table with one or more non-integer values"
                return
            end
        end
        return bridged.decode(_python.builtins.list(tok))
    end

    ---@return nil
    function kobold.halt_generation()
        generating = false
    end


    --==========================================================================
    -- Core script API
    --==========================================================================

    koboldbridge.userscripts = {}  ---@type table<integer, string>
    koboldbridge.num_userscripts = 0
    koboldbridge.inmod = nil  ---@type function|nil
    koboldbridge.genmod = nil  ---@type function|nil
    koboldbridge.outmod = nil  ---@type function|nil

    ---@class KoboldUserScript
    ---@field inmod function|nil
    ---@field genmod function|nil
    ---@field outmod function|nil

    ---@class KoboldCoreScript
    ---@field inmod function|nil
    ---@field genmod function|nil
    ---@field outmod function|nil


    ----------------------------------------------------------------------------

    ---@class KoboldUserScriptModule
    ---@field filename string
    ---@field modulename string
    ---@field description string
    ---@field inmod function|nil
    ---@field genmod function|nil
    ---@field outmod function|nil
    local KoboldUserScriptModule = setmetatable({
        _name = "KoboldUserScriptModule",
    }, metawrapper)
    local KoboldUserScriptModule_mt = setmetatable({}, metawrapper)

    local KoboldUserScriptModule_fields = {
        filename = false,
        modulename = false,
        description = false,
        inmod = false,
        genmod = false,
        outmod = false,
    }

    ---@generic K
    ---@param t KoboldUserScriptModule
    ---@param k K
    ---@return K, any
    function KoboldUserScriptModule_mt._kobold_next(t, k)
        k = (next(KoboldUserScriptModule_fields, k))
        return k, t[k]
    end

    ---@param t KoboldUserScriptModule
    ---@return function, KoboldUserScriptModule, nil
    function KoboldUserScriptModule_mt.__pairs(t)
        return next, t, nil
    end

    ---@param t KoboldUserScriptModule
    function KoboldUserScriptModule_mt.__index(t, k)
        if type(k) == "string" and KoboldUserScriptModule_fields[k] ~= nil then
            return rawget(t, "_" .. k)
        end
        return rawget(t, k)
    end

    ---@param t KoboldUserScriptModule
    function KoboldUserScriptModule_mt.__newindex(t, k, v)
        error("`"..rawget(t, "_name").."` is a read-only class")
    end


    ----------------------------------------------------------------------------

    ---@class KoboldUserScriptList_base
    ---@type table<integer, KoboldUserScriptModule>
    local _ = {}

    ---@class KoboldUserScriptList : KoboldUserScriptList_base
    local KoboldUserScriptList = setmetatable({
        _name = "KoboldUserScriptList",
    }, metawrapper)
    local KoboldUserScriptList_mt = setmetatable({}, metawrapper)

    ---@param t KoboldUserScriptList
    ---@return integer
    function KoboldUserScriptList_mt.__len(t)
        return koboldbridge.num_userscripts
    end

    ---@param t KoboldUserScriptList
    ---@param k integer
    ---@return KoboldUserScriptModule|nil
    function KoboldUserScriptList_mt.__index(t, k)
        if type(k) == "number" and math.tointeger(k) ~= nil then
            return koboldbridge.userscripts[k]
        end
    end

    ---@generic K
    ---@param t KoboldUserScriptList
    ---@param k K
    ---@return K, any
    function KoboldUserScriptList_mt._kobold_next(t, k)
        if k == nil then
            k = 0
        elseif type(k) ~= "number" then
            return nil
        end
        k = k + 1
        local v = t[k]
        if v == nil then
            return nil
        end
        return v.filename, v
    end

    ---@param t KoboldUserScriptList
    ---@return function, KoboldUserScriptList, nil
    function KoboldUserScriptList_mt.__pairs(t)
        return next, t, nil
    end

    ---@param t KoboldUserScriptList
    function KoboldUserScriptList_mt.__newindex(t, k, v)
        error("`"..rawget(t, "_name").."` is a read-only class")
    end


    ----------------------------------------------------------------------------

    ---@param t KoboldCoreLib
    ---@return string
    function KoboldCoreLib_getters.userscripts(t)
        return koboldbridge.userscripts
    end

    ---@param t KoboldCoreLib
    ---@param v string
    ---@return KoboldCoreLib
    function KoboldCoreLib_setters.userscripts(t, v)
        error("`KoboldCoreLib.userscripts` is a read-only attribute")
    end


    --==========================================================================
    -- Sandboxing code
    --==========================================================================

    local envs = {}

    local old_load = load
    local function safe_load(chunk, chunkname, mode, env)
        if mode == nil then
            mode = "t"
        elseif mode ~= "t" then
            error("Calling `load` with a `mode` other than 't' is disabled for security reasons")
            return
        end
        if env == nil then
            env = _G
        end
        return old_load(chunk, chunkname, mode, env)
    end

    local old_loadfile = loadfile
    local old_package_loaded = package.loaded
    local old_package_searchers = package.searchers
    ---@param modname string
    ---@param env table<string, any>
    ---@return any, string|nil
    local function safe_require_with_env(modname, env)
        if modname == "bridge" then
            return function() return _G.kobold, _G.koboldcore end
        end
        if type(modname) == "number" then
            modname = tostring(modname)
        elseif type(modname) ~= "string" then
            error("bad argument #1 to 'require' (string expected, got "..type(modname)..")")
            return
        end
        local allowsearch = type(modname) == "string" and string.match(modname, "[^%w._]") == nil and string.match(modname, "%.%.") == nil
        if allowsearch and old_package_loaded[modname] then
            return old_package_loaded[modname]
        end
        local loader, path
        local errors = {}
        local n_errors = 0
        for k, v in ipairs(old_package_searchers) do
            loader, path = v(modname)
            if allowsearch and type(loader) == "function" then
                break
            elseif type(loader) == "string" then
                n_errors = n_errors + 1
                errors[n_errors] = "\n\t" .. loader
            end
        end
        if not allowsearch or type(loader) ~= "function" then
            error("module '" .. modname .. "' not found:" .. table.concat(errors))
            return
        end
        local retval = old_loadfile(path, "t", env)()
        old_package_loaded[modname] = retval == nil or retval
        return old_package_loaded[modname], path
    end
    ---@param modname string
    ---@return any, string|nil
    local function safe_require(modname)
        return safe_require_with_env(modname, _G)
    end

    local sandbox_template_env = {
        assert = assert,
        connectgarbage = collectgarbage,
        error = error,
        getmetatable = getmetatable,
        ipairs = ipairs,
        load = safe_load,
        next = next,
        pairs = pairs,
        pcall = pcall,
        print = print,
        rawequal = rawequal,
        rawget = rawget,
        rawlen = rawlen,
        rawset = rawset,
        select = select,
        setmetatable = setmetatable,
        tonumber = tonumber,
        tostring = tostring,
        type = type,
        _VERSION = _VERSION,
        warn = warn,
        xpcall = xpcall,
        coroutine = {
            close = coroutine.close,
            create = coroutine.create,
            isyieldable = coroutine.isyieldable,
            resume = coroutine.resume,
            running = coroutine.running,
            status = coroutine.status,
            wrap = coroutine.wrap,
            yield = coroutine.yield,
        },
        require = safe_require,
        package = {
            config = package.config,
        },
        string = {
            byte = string.byte,
            char = string.char,
            dump = string.dump,
            find = string.find,
            format = string.format,
            gmatch = string.gmatch,
            gsub = string.gsub,
            len = string.len,
            lower = string.lower,
            match = string.match,
            pack = string.pack,
            packsize = string.packsize,
            rep = string.rep,
            reverse = string.reverse,
            sub = string.sub,
            unpack = string.unpack,
            upper = string.upper,
        },
        utf8 = {
            char = utf8.char,
            charpattern = utf8.charpattern,
            codes = utf8.codes,
            codepoint = utf8.codepoint,
            len = utf8.len,
            offset = utf8.offset,
        },
        table = {
            concat = table.concat,
            insert = table.insert,
            move = table.move,
            pack = table.pack,
            remove = table.remove,
            sort = table.sort,
            unpack = table.unpack,
        },
        math = {
            abs = math.abs,
            acos = math.acos,
            asin = math.asin,
            atan = math.atan,
            atan2 = math.atan2,
            ceil = math.ceil,
            cos = math.cos,
            cosh = math.cosh,
            deg = math.deg,
            exp = math.exp,
            floor = math.floor,
            fmod = math.fmod,
            frexp = math.frexp,
            huge = math.huge,
            ldexp = math.ldexp,
            log = math.log,
            log10 = math.log10,
            max = math.max,
            maxinteger = math.maxinteger,
            min = math.min,
            mininteger = math.mininteger,
            modf = math.modf,
            pi = math.pi,
            pow = math.pow,
            rad = math.rad,
            random = math.random,
            randomseed = function() warn("WARNING: math.randomseed() is not permitted; please use the mt19937ar library instead") end,
            sin = math.sin,
            sinh = math.sinh,
            sqrt = math.sqrt,
            tan = math.tan,
            tanh = math.tanh,
            tointeger = math.tointeger,
            type = math.type,
            ult = math.ult,
        },
        io = {
            read = io.read,
            write = io.write,
            flush = io.flush,
            type = io.type,
        },
        os = {
            clock = os.clock,
            date = os.date,
            difftime = os.difftime,
            exit = function() end,
            getenv = os.getenv,
            time = os.time,
            tmpname = os.tmpname,
        },
        debug = {
            getinfo = debug.getinfo,
            gethook = debug.gethook,
            getmetatable = debug.getmetatable,
            getuservalue = debug.getuservalue,
            sethook = debug.sethook,
            setmetatable = debug.setmetatable,
            setuservalue = debug.setuservalue,
            traceback = debug.traceback,
            upvalueid = debug.upvalueid,
        },
    }

    function koboldbridge.get_universe(universe)
        local env = envs[universe]
        if env == nil then
            envs[universe] = deepcopy(sandbox_template_env)
            env = envs[universe]
            envs[universe].kobold = deepcopy(kobold)
            if universe == 0 then
                envs[universe].koboldcore = deepcopy(koboldcore)
            end
            env._G = env
        end
        return env
    end

    function koboldbridge.obliterate_multiverse()
        envs = {}
    end


    --==========================================================================
    -- Connection to aiserver.py
    --==========================================================================

    ---@return nil
    function koboldbridge.load_userscripts(filenames, modulenames, descriptions)
        local old_path = set_require_path(bridged.userscript_path)
        koboldbridge.userscripts = {}
        koboldbridge.num_userscripts = 0
        for i, filename in _python.enumerate(filenames) do
            bridged.load_callback(filename)
            local _userscript = safe_require_with_env(filename, koboldbridge.get_universe(filename))
            local userscript = deepcopy(KoboldUserScriptModule)
            rawset(userscript, "_inmod", _userscript.inmod)
            rawset(userscript, "_genmod", _userscript.genmod)
            rawset(userscript, "_outmod", _userscript.outmod)
            rawset(userscript, "_filename", filename)
            rawset(userscript, "_modulename", modulenames[i])
            rawset(userscript, "_description", descriptions[i])
            koboldbridge.userscripts[i+1] = userscript
            koboldbridge.num_userscripts = i + 1
        end
        set_require_path(old_path)
    end

    ---@return nil
    function koboldbridge.load_corescript(filename)
        set_require_path(bridged.corescript_path)
        local corescript = safe_require_with_env(filename, koboldbridge.get_universe(0))
        koboldbridge.inmod = corescript.inmod
        koboldbridge.genmod = corescript.genmod
        koboldbridge.outmod = corescript.outmod
        set_require_path(bridged.lib_path)
    end

    ---@return any, any, any
    function koboldbridge.execute()
        local i, g, o
        local num_generated = 0
        g = {}
        generating = true
        userstate = "inmod"
        if koboldbridge.inmod ~= nil then
            i = koboldbridge.inmod()
        end
        userstate = "genmod"
        bridged.start_generation()
        while generating and num_generated < bridged.get_gen_len() do
            bridged.generation_step()
            if koboldbridge.genmod ~= nil then
                table.insert(g, koboldbridge.genmod())
                if genmod_comparison_context ~= kobold.worldinfo:compute_context() then
                    bridged.register_context_change()
                    genmod_comparison_context = nil
                end
            end
            num_generated = num_generated + 1
        end
        userstate = "outmod"
        bridged.stop_generation()
        if koboldbridge.outmod ~= nil then
            o = koboldbridge.outmod()
        end
        generating = true
        userstate = "inmod"
        return i, g, o
    end


    ----------------------------------------------------------------------------

    metawrapper.__newindex = nil
    setmetatable(KoboldWorldInfoEntry, KoboldWorldInfoEntry_mt)
    setmetatable(KoboldWorldInfoFolder, KoboldWorldInfoFolder_mt)
    setmetatable(KoboldWorldInfoFolderSelector, KoboldWorldInfoFolderSelector_mt)
    setmetatable(KoboldWorldInfo, KoboldWorldInfo_mt)
    setmetatable(KoboldUserScriptModule, KoboldUserScriptModule_mt)
    setmetatable(KoboldUserScriptList, KoboldUserScriptList_mt)
    setmetatable(kobold, KoboldLib_mt)
    setmetatable(koboldcore, KoboldCoreLib_mt)

    return kobold, koboldcore, koboldbridge
end
