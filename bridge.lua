-- KoboldAI Lua 5.4 Bridge


---@param _python? table<string, any>
---@param _bridged? table<string, any>
---@return KoboldLib, KoboldCoreLib?
return function(_python, _bridged)

    --==========================================================================
    -- Globally allows using a _kobold_next metamethod for "Kobold" classes only
    --==========================================================================

    local old_next = next
    ---@generic K, V
    ---@param t table<K, V>
    ---@param k? K
    ---@return K?, V?
    function next(t, k)
        local meta = getmetatable(t)
        return ((meta ~= nil and type(rawget(t, "_name")) == "string" and string.match(rawget(t, "_name"), "^Kobold") and type(meta._kobold_next) == "function") and meta._kobold_next or old_next)(t, k)
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

    ---@param paths string|table<integer, string>
    ---@return nil
    local function set_require_path(paths)
        if type(paths) == "string" then
            paths = {paths}
        end
        local config = {}
        local i = 1
        for substring in string.gmatch(package.config, "[^\n]+") do
            config[i] = substring
            i = i + 1
        end
        local _paths = {}
        for i, path in ipairs(paths) do
            _paths[i] = path .. config[1] .. config[3] .. ".lua" .. config[2] .. path .. config[1] .. config[3] .. config[1] .. "init.lua"
        end
        package.path = table.concat(_paths, config[2])
        package.cpath = ""
    end

    ---@param path string
    ---@param filename string
    ---@return string
    local function join_folder_and_filename(path, filename)
        return path .. string.match(package.config, "[^\n]+") .. filename
    end


    --==========================================================================
    -- _bridged preprocessing
    --==========================================================================

    local bridged = {}
    for k in _python.iter(_bridged) do
        local v = _bridged[k]
        bridged[k] = type(v) == "userdata" and _python.as_attrgetter(v) or v
    end
    set_require_path(bridged.lib_paths)


    --==========================================================================
    -- Wraps most functions in this file so that they restore original
    -- metatables prior to executing the function body
    --==========================================================================

    local wrapped = false

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
    function metawrapper.__newindex(t, k, wrapped_func)
        if type(wrapped_func) == "function" then
            return rawset(t, k, function(...)
                local _needs_unwrap = false
                if not wrapped then
                    metatables:overwrite()
                    metatables_original:restore()
                    _needs_unwrap = true
                    wrapped = true
                end
                local r = table.pack(wrapped_func(...))
                if _needs_unwrap then
                    metatables:restore()
                    wrapped = false
                end
                return table.unpack(r, 1, r.n)
            end)
        else
            return rawset(t, k, wrapped_func)
        end
    end


    --==========================================================================
    -- Modules
    --==========================================================================

    ---@class KoboldLib
    ---@field API_VERSION number
    ---@field authorsnote string
    ---@field authorsnotetemplate string
    ---@field memory string
    ---@field submission string
    ---@field model string
    ---@field modeltype "'readonly'"|"'api'"|"'unknown'"|"'gpt2'"|"'gpt2-medium'"|"'gpt2-large'"|"'gpt2-xl'"|"'gpt-neo-125M'"|"'gpt-neo-1.3B'"|"'gpt-neo-2.7B'"|"'gpt-j-6B'"
    ---@field modelbackend "'readonly'"|"'api'"|"'transformers'"|"'mtj'"
    ---@field is_custommodel boolean
    ---@field custmodpth string
    ---@field logits table<integer, table<integer, number>>
    ---@field logits_rows integer
    ---@field logits_cols integer
    ---@field generated table<integer, table<integer, integer>>
    ---@field generated_rows integer
    ---@field generated_cols integer
    ---@field outputs table<integer, string>
    ---@field num_outputs integer
    ---@field feedback string
    ---@field is_config_file_open boolean
    local kobold = setmetatable({API_VERSION = 1.2}, metawrapper)
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
    local koboldbridge = {}

    koboldbridge.regeneration_required = false
    koboldbridge.resend_settings_required = false
    koboldbridge.generating = true
    koboldbridge.restart_sequence = nil
    koboldbridge.userstate = nil
    koboldbridge.logits = {}
    koboldbridge.vocab_size = 0
    koboldbridge.generated = {}
    koboldbridge.generated_cols = 0
    koboldbridge.outputs = {}
    koboldbridge.feedback = nil  ---@type string?

    function koboldbridge:clear_userscript_metadata()
        self.logging_name = nil
        self.filename = nil
    end

    ---@return nil
    local function maybe_require_regeneration()
        if koboldbridge.userstate == "genmod" or koboldbridge.userstate == "outmod" then
            koboldbridge.regeneration_required = true
        end
    end


    --==========================================================================
    -- Userscript API: Configuration
    --==========================================================================

    local config_files = {}  ---@type table<string, file*>
    local config_file_filename_map = {}  ---@type table<file*, string>

    ---@return file*?
    local function open_and_handle_errors(...)
        local file, err_msg = io.open(...)
        if err_msg ~= nil then
            koboldbridge.obliterate_multiverse()
            error(err_msg)
            return
        end
        return file
    end

    ---@param file? file*
    local function new_close_pre(file)
        if file == nil then
            file = io.output()
        end
        local filename = config_file_filename_map[file]
        if filename ~= nil then
            config_file_filename_map[file] = nil
            config_files[filename] = nil
        end
    end

    ---@param f fun(file?: file*)
    local function _new_close(f)
        ---@param file? file*
        return function(file)
            new_close_pre(file)
            return f(file)
        end
    end
    debug.getmetatable(io.stdout).__index.close = _new_close(io.stdout.close)
    debug.getmetatable(io.stdout).__close = _new_close(io.stdout.close)

    ---@param filename string
    ---@return boolean
    local function is_config_file_open(filename)
        return config_files[filename] ~= nil
    end

    ---@param filename string
    ---@param clear? boolean
    ---@return file*
    local function get_config_file(filename, clear)
        if not is_config_file_open(filename) then
            local config_filepath = join_folder_and_filename(bridged.config_path, filename .. ".conf")
            open_and_handle_errors(config_filepath, "a"):close()
            config_files[filename] = open_and_handle_errors(config_filepath, clear and "w+b" or "r+b")
            config_file_filename_map[config_files[filename]] = filename
        end
        return config_files[filename]
    end

    ---@param clear? boolean
    ---@return file*
    function kobold.get_config_file(clear)
        return get_config_file(koboldbridge.filename, clear)
    end

    ---@param t KoboldLib
    ---@return boolean
    function KoboldLib_getters.is_config_file_open(t)
        return is_config_file_open(koboldbridge.filename)
    end

    ---@param t KoboldLib
    ---@param v boolean
    function KoboldLib_setters.is_config_file_open(t, v)
        error("`KoboldLib.is_config_file_open` is a read-only attribute")
    end


    --==========================================================================
    -- Userscript API: World Info
    --==========================================================================

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
        return _python.as_attrgetter(bridged.vars.worldinfo_u).get(rawget(self, "_uid")) ~= nil
    end

    ---@param submission? string
    ---@param kwargs? table<string, any>
    ---@return string
    function KoboldWorldInfoEntry:compute_context(submission, kwargs)
        if not check_validity(self) then
            return ""
        elseif submission == nil then
            submission = kobold.submission
        elseif type(submission) ~= "string" then
            error("`compute_context` takes a string or nil as argument #1, but got a " .. type(submission))
            return ""
        end
        return bridged.compute_context(submission, {self.uid}, nil, kwargs)
    end

    ---@generic K
    ---@param t KoboldWorldInfoEntry|KoboldWorldInfoFolder|KoboldWorldInfo|KoboldWorldInfoFolderSelector
    ---@param k K
    ---@return K, any
    function KoboldWorldInfoEntry_mt._kobold_next(t, k)
        local _t = fields[rawget(t, "_name")]
        if _t == nil then
            return
        end
        return next(_t, k)
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
                if k ~= "comment" and not (t.selective and k == "keysecondary") then
                    maybe_require_regeneration()
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
    ---@field name string
    local KoboldWorldInfoFolder = setmetatable({
        _name = "KoboldWorldInfoFolder",
    }, metawrapper)

    fields.KoboldWorldInfoFolder = {
        "uid",
    }
    local KoboldWorldInfoFolder_mt = setmetatable({}, metawrapper)

    ---@param u integer
    ---@return KoboldWorldInfoEntry?
    function KoboldWorldInfoFolder:finduid(u)
        if not check_validity(self) or type(u) ~= "number" then
            return
        end
        local query = _python.as_attrgetter(bridged.vars.worldinfo_u).get(u)
        if query == nil or (rawget(self, "_name") == "KoboldWorldInfoFolder" and self.uid ~= _python.as_attrgetter(query).get("folder")) then
            return
        end
        local entry = deepcopy(KoboldWorldInfoEntry)
        rawset(entry, "_uid", u)
        return entry
    end

    ---@param submission? string
    ---@param entries? KoboldWorldInfoEntry|table<any, KoboldWorldInfoEntry>
    ---@param kwargs? table<string, any>
    ---@return string
    function KoboldWorldInfoFolder:compute_context(submission, entries, kwargs)
        if not check_validity(self) then
            return ""
        elseif submission == nil then
            submission = kobold.submission
        elseif type(submission) ~= "string" then
            error("`compute_context` takes a string or nil as argument #1, but got a " .. type(submission))
            return ""
        end
        local _entries
        if entries ~= nil then
            if type(entries) ~= "table" or (entries.name ~= nil and entries.name ~= "KoboldWorldInfoEntry") then
                error("`compute_context` takes a KoboldWorldInfoEntry, table of KoboldWorldInfoEntries or nil as argument #2, but got a " .. type(entries))
                return ""
            elseif entries.name == "KoboldWorldInfoEntry" then
                _entries = {entries}
            else
                _entries = {}
                for k, v in pairs(entries) do
                    if type(v) == "table" and v.name == "KoboldWorldInfoEntry" and v:is_valid() then
                        _entries[k] = v.uid
                    end
                end
            end
        end
        local folders
        if self.name == "KoboldWorldInfoFolder" then
            folders = {rawget(self, "_uid")}
        end
        return bridged.compute_context(submission, _entries, folders, kwargs)
    end

    ---@return boolean
    function KoboldWorldInfoFolder:is_valid()
        return _python.as_attrgetter(bridged.vars.wifolders_d).get(rawget(self, "_uid")) ~= nil
    end

    ---@param t KoboldWorldInfoFolder
    ---@return integer
    function KoboldWorldInfoFolder_mt.__len(t)
        if not check_validity(t) then
            return 0
        end
        return math.tointeger(_python.builtins.len(_python.as_attrgetter(bridged.vars.wifolders_u).get(t.uid))) - 1
    end

    KoboldWorldInfoFolder_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfoFolder_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    ---@param t KoboldWorldInfoFolder|KoboldWorldInfo
    ---@return KoboldWorldInfoEntry?
    function KoboldWorldInfoFolder_mt.__index(t, k)
        if not check_validity(t) then
            return
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "uid" then
            return rawget(t, "_uid")
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "name" then
            return bridged.folder_get_attr(t.uid, k)
        elseif type(k) == "number" then
            local query = rawget(t, "_name") == "KoboldWorldInfoFolder" and _python.as_attrgetter(bridged.vars.wifolders_u).get(t.uid) or bridged.vars.worldinfo_i
            k = math.tointeger(k)
            if k == nil or k < 1 or k > #t then
                return
            end
            local entry = deepcopy(KoboldWorldInfoEntry)
            rawset(entry, "_uid", math.tointeger(query[k-1].uid))
            return entry
        end
    end

    ---@param t KoboldWorldInfoFolder|KoboldWorldInfo
    ---@return KoboldWorldInfoFolder|KoboldWorldInfo
    function KoboldWorldInfoFolder_mt.__newindex(t, k, v)
        if not check_validity(t) then
            return
        elseif type(k) == "number" and math.tointeger(k) ~= nil then
            error("Cannot write to integer indices of `"..rawget(t, "_name").."`")
        elseif rawget(t, "_name") == "KoboldWorldInfoFolder" and k == "uid" then
            error("`"..rawget(t, "_name").."."..k.."` is a read-only attribute")
        elseif t == "name" then
            if type(v) ~= "string" then
                error("`"..rawget(t, "_name").."."..k.."` must be a string; you attempted to set it to a "..type(v))
                return
            end
            bridged.folder_set_attr(t.uid, k, v)
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
    ---@return KoboldWorldInfoFolder?
    function KoboldWorldInfoFolderSelector:finduid(u)
        if not check_validity(self) or type(u) ~= "number" then
            return
        end
        local query = _python.as_attrgetter(bridged.vars.wifolders_d).get(u)
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
        return _python.builtins.len(bridged.vars.wifolders_l)
    end

    KoboldWorldInfoFolderSelector_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfoFolderSelector_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    ---@param t KoboldWorldInfoFolderSelector
    ---@return KoboldWorldInfoFolder?
    function KoboldWorldInfoFolderSelector_mt.__index(t, k)
        if not check_validity(t) or type(k) ~= "number" or math.tointeger(k) == nil or k < 1 or k > #t then
            return
        end
        local folder = deepcopy(KoboldWorldInfoFolder)
        rawset(folder, "_uid", math.tointeger(bridged.vars.wifolders_l[k-1]))
        return folder
    end

    ---@param t KoboldWorldInfoFolderSelector
    ---@return KoboldWorldInfoFolderSelector
    function KoboldWorldInfoFolderSelector_mt.__newindex(t, k, v)
        if check_validity(t) or (type(k) == "number" and math.tointeger(k) ~= nil) then
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
        return math.tointeger(_python.builtins.len(bridged.vars.worldinfo)) - math.tointeger(_python.builtins.len(bridged.vars.wifolders_l)) - 1
    end

    KoboldWorldInfo_mt._kobold_next = KoboldWorldInfoEntry_mt._kobold_next

    KoboldWorldInfo_mt.__pairs = KoboldWorldInfoEntry_mt.__pairs

    KoboldWorldInfo_mt.__index = KoboldWorldInfoFolder_mt.__index

    KoboldWorldInfo_mt.__newindex = KoboldWorldInfoFolder_mt.__newindex

    kobold.worldinfo = KoboldWorldInfo


    --==========================================================================
    -- Userscript API: Story chunks
    --==========================================================================

    ---@class KoboldStoryChunk
    ---@field num integer
    ---@field content string
    local KoboldStoryChunk = setmetatable({
        _name = "KoboldStoryChunk",
    }, metawrapper)
    local KoboldStoryChunk_mt = setmetatable({}, metawrapper)

    local KoboldStoryChunk_fields = {
        num = false,
        content = false,
    }

    ---@generic K
    ---@param t KoboldStoryChunk
    ---@param k K
    ---@return K, any
    function KoboldStoryChunk_mt._kobold_next(t, k)
        k = (next(KoboldStoryChunk_fields, k))
        return k, t[k]
    end

    ---@param t KoboldStoryChunk
    ---@return function, KoboldStoryChunk, nil
    function KoboldStoryChunk_mt.__pairs(t)
        return next, t, nil
    end

    ---@param t KoboldStoryChunk
    function KoboldStoryChunk_mt.__index(t, k)
        if k == "num" then
            return rawget(t, "_num")
        end
        if k == "content" then
            if rawget(t, "_num") == 0 then
                if bridged.vars.gamestarted then
                    local prompt = koboldbridge.userstate == "genmod" and bridged.vars._prompt or bridged.vars.prompt
                    return prompt
                end
            end
            local actions = koboldbridge.userstate == "genmod" and bridged.vars._actions or bridged.vars.actions
            return _python.as_attrgetter(actions).get(math.tointeger(rawget(t, "_num")) - 1)
        end
    end

    ---@param t KoboldStoryChunk
    function KoboldStoryChunk_mt.__newindex(t, k, v)
        if k == "num" then
            error("`"..rawget(t, "_name").."."..k.."` is a read-only attribute")
            return
        elseif k == "content" then
            if type(v) ~= "string" then
                error("`"..rawget(t, "_name").."."..k.."` must be a string; you attempted to set it to a "..type(v))
                return
            end
            local _k = math.tointeger(rawget(t, "_num"))
            if _k == nil or _k < 0 then
                return
            elseif _k == 0 and v == "" then
                error("Attempted to set the prompt chunk's content to the empty string; this is not allowed")
                return
            end
            local actions = koboldbridge.userstate == "genmod" and bridged.vars._actions or bridged.vars.actions
            if _k ~= 0 and _python.as_attrgetter(actions).get(_k-1) == nil then
                return
            end
            bridged.set_chunk(_k, v)
            maybe_require_regeneration()
            return t
        end
    end


    ----------------------------------------------------------------------------

    ---@class KoboldStory_base
    ---@type table<integer, KoboldStoryChunk>
    local _ = {}

    ---@class KoboldStory : KoboldStory_base
    local KoboldStory = setmetatable({
        _name = "KoboldStory",
    }, metawrapper)
    local KoboldStory_mt = setmetatable({}, metawrapper)

    ---@return fun(): KoboldStoryChunk, table, nil
    function KoboldStory:forward_iter()
        local actions = koboldbridge.userstate == "genmod" and bridged.vars._actions or bridged.vars.actions
        local nxt, iterator = _python.iter(actions)
        local run_once = false
        local function f()
            if not bridged.vars.gamestarted then
                return
            end
            local chunk = deepcopy(KoboldStoryChunk)
            local _k
            if not run_once then
                _k = -1
                run_once = true
            else
                _k = nxt(iterator)
            end
            if _k == nil then
                return
            else
                _k = math.tointeger(_k) + 1
            end
            rawset(chunk, "_num", _k)
            return chunk
        end
        return f, {}, nil
    end

    ---@return fun(): KoboldStoryChunk, table, nil
    function KoboldStory:reverse_iter()
        local actions = koboldbridge.userstate == "genmod" and bridged.vars._actions or bridged.vars.actions
        local nxt, iterator = _python.iter(_python.builtins.reversed(actions))
        local last_run = false
        local function f()
            if not bridged.vars.gamestarted or last_run then
                return
            end
            local chunk = deepcopy(KoboldStoryChunk)
            local _k = nxt(iterator)
            if _k == nil then
                _k = 0
                last_run = true
            else
                _k = math.tointeger(_k) + 1
            end
            rawset(chunk, "_num", _k)
            return chunk
        end
        return f, {}, nil
    end

    ---@param t KoboldStory
    function KoboldStory_mt.__pairs(t)
        return function() return nil end, t, nil
    end

    ---@param t KoboldStory
    function KoboldStory_mt.__index(t, k)
        if type(k) == "number" and math.tointeger(k) ~= nil then
            local chunk = deepcopy(KoboldStoryChunk)
            rawset(chunk, "_num", math.tointeger(k))
            if chunk.content == nil then
                return nil
            end
            return chunk
        end
    end

    ---@param t KoboldStory
    function KoboldStory_mt.__newindex(t, k, v)
        error("`"..rawget(t, "_name").."` is a read-only class")
    end

    kobold.story = KoboldStory


    --==========================================================================
    -- Userscript API: Settings
    --==========================================================================

    ---@class KoboldSettings_base
    ---@type table<string, any>
    local _ = {}

    ---@class KoboldSettings : KoboldSettings_base
    ---@field numseqs integer
    ---@field genamt integer
    ---@field anotedepth integer
    ---@field settemp number
    ---@field settopp number
    ---@field settopk integer
    ---@field settfs number
    ---@field settypical number
    ---@field settopa number
    ---@field setreppen number
    ---@field setreppenslope number
    ---@field setreppenrange number
    ---@field settknmax integer
    ---@field setwidepth integer
    ---@field setuseprompt boolean
    ---@field setadventure boolean
    ---@field setdynamicscan boolean
    ---@field setnopromptgen boolean
    ---@field setrngpersist boolean
    ---@field temp number
    ---@field topp number
    ---@field topk integer
    ---@field top_p number
    ---@field top_k integer
    ---@field tfs number
    ---@field typical number
    ---@field topa number
    ---@field reppen number
    ---@field reppenslope number
    ---@field reppenrange number
    ---@field tknmax integer
    ---@field widepth integer
    ---@field useprompt boolean
    ---@field adventure boolean
    ---@field dynamicscan boolean
    ---@field nopromptgen boolean
    ---@field rngpersist boolean
    ---@field frmttriminc boolean
    ---@field frmtrmblln boolean
    ---@field frmtrmspch boolean
    ---@field frmtadsnsp boolean
    ---@field frmtsingleline boolean
    ---@field triminc boolean
    ---@field rmblln boolean
    ---@field rmspch boolean
    ---@field adsnsp boolean
    ---@field singleline boolean
    ---@field chatmode boolean
    ---@field chatname string
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
        if k == "genamt" or k == "output" or k == "setoutput" then
            return math.tointeger(bridged.get_genamt()), true
        elseif k == "numseqs" or k == "numseq" or k == "setnumseq" then
            return math.tointeger(bridged.get_numseqs()), true
        elseif bridged.has_setting(k) then
            return bridged.get_setting(k), true
        else
            return nil, false
        end
    end

    ---@param t KoboldSettings_base
    function KoboldSettings_mt.__newindex(t, k, v)
        if (k == "genamt" or k == "output" or k == "setoutput") and type(v) == "number" and math.tointeger(v) ~= nil and v >= 0 then
            bridged.set_genamt(v)
            koboldbridge.resend_settings_required = true
        elseif (k == "numseqs" or k == "numseq" or k == "setnumseq") and type(v) == "number" and math.tointeger(v) ~= nil and v >= 1 then
            if koboldbridge.userstate == "genmod" then
                error("Cannot set numseqs from a generation modifier")
                return
            end
            bridged.set_numseqs(v)
            koboldbridge.resend_settings_required = true
        elseif type(k) == "string" and bridged.has_setting(k) and type(v) == type(bridged.get_setting(k)) then
            if bridged.set_setting(k, v) == true then
                maybe_require_regeneration()
            end
            koboldbridge.resend_settings_required = true
        end
        return t
    end

    kobold.settings = KoboldSettings


    --==========================================================================
    -- Userscript API: Memory / Author's Note
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
        if type(v) ~= "string" then
            error("`KoboldLib.memory` must be a string; you attempted to set it to a "..type(v))
            return
        end
        maybe_require_regeneration()
        bridged.set_memory(v)
    end

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.authorsnote(t)
        return bridged.get_authorsnote()
    end

    ---@param t KoboldLib
    ---@param v string
    ---@return KoboldLib
    function KoboldLib_setters.authorsnote(t, v)
        if type(v) ~= "string" then
            error("`KoboldLib.authorsnote` must be a string; you attempted to set it to a "..type(v))
            return
        end
        maybe_require_regeneration()
        bridged.set_authorsnote(v)
    end

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.authorsnotetemplate(t)
        return bridged.get_authorsnotetemplate()
    end

    ---@param t KoboldLib
    ---@param v string
    ---@return KoboldLib
    function KoboldLib_setters.authorsnotetemplate(t, v)
        if type(v) ~= "string" then
            error("`KoboldLib.authorsnotetemplate` must be a string; you attempted to set it to a "..type(v))
            return
        end
        maybe_require_regeneration()
        bridged.set_authorsnotetemplate(v)
    end


    --==========================================================================
    -- Userscript API: User-submitted text (after applying input formatting)
    --==========================================================================

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.submission(t)
        return bridged.vars.submission
    end

    ---@param t KoboldLib
    ---@param v string
    function KoboldLib_setters.submission(t, v)
        if koboldbridge.userstate ~= "inmod" then
            error("Cannot write to `KoboldLib.submission` from outside of an input modifier")
            return
        elseif type(v) ~= "string" then
            error("`KoboldLib.submission` must be a string; you attempted to set it to a " .. type(v))
            return
        elseif not bridged.vars.gamestarted and v == "" then
            error("`KoboldLib.submission` must not be set to the empty string when the story is empty")
            return
        end
        bridged.vars.submission = v
    end


    --==========================================================================
    -- Userscript API: Soft prompt
    --==========================================================================

    ---@param t KoboldLib
    ---@return string?
    function KoboldLib_getters.spfilename(t)
        return bridged.get_spfilename()
    end

    ---@param t KoboldLib
    ---@param v string?
    function KoboldLib_setters.spfilename(t, v)
        if v:find("/") or v:find("\\") then
            error("Cannot set `KoboldLib.spfilename` to a string that contains slashes")
        end
        if bridged.set_spfilename(v) then
            maybe_require_regeneration()
        end
    end


    --==========================================================================
    -- Userscript API: Model information
    --==========================================================================

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.modeltype(t)
        return bridged.get_modeltype()
    end

    ---@param t KoboldLib
    ---@param v string
    function KoboldLib_setters.modeltype(t, v)
        error("`KoboldLib.modeltype` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.model(t)
        return bridged.vars.model
    end

    ---@param t KoboldLib
    ---@param v string
    function KoboldLib_setters.model(t, v)
        error("`KoboldLib.model` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.modelbackend(t)
        return bridged.get_modelbackend()
    end

    ---@param t KoboldLib
    ---@param v string
    function KoboldLib_setters.modelbackend(t, v)
        error("`KoboldLib.modelbackend` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return boolean
    function KoboldLib_getters.is_custommodel(t)
        return bridged.is_custommodel()
    end

    ---@param t KoboldLib
    ---@param v boolean
    function KoboldLib_setters.is_custommodel(t, v)
        error("`KoboldLib.is_custommodel` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return string
    function KoboldLib_getters.custmodpth(t)
        return bridged.vars.custmodpth
    end

    ---@param t KoboldLib
    ---@param v string
    function KoboldLib_setters.custmodpth(t, v)
        error("`KoboldLib.custmodpth` is a read-only attribute")
    end


    --==========================================================================
    -- Userscript API: Logit Warping
    --==========================================================================

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_getters.logits_rows(t)
        if koboldbridge.userstate ~= "genmod" then
            return 0
        end
        local backend = kobold.modelbackend
        if backend == "readonly" or backend == "api" then
            return 0
        end
        return kobold.settings.numseqs
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_setters.logits_rows(t)
        error("`KoboldLib.logits_rows` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_getters.logits_cols(t)
        if koboldbridge.userstate ~= "genmod" then
            return 0
        end
        local backend = kobold.modelbackend
        if backend == "readonly" or backend == "api" then
            return 0
        end
        return math.tointeger(koboldbridge.vocab_size)
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_setters.logits_cols(t)
        error("`KoboldLib.logits_cols` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return table<integer, table<integer, number>>
    function KoboldLib_getters.logits(t)
        if koboldbridge.userstate ~= "genmod" then
            return
        end
        return koboldbridge.logits
    end

    ---@param t KoboldLib
    ---@param v table<integer, table<integer, number>>
    function KoboldLib_setters.logits(t, v)
        if koboldbridge.userstate ~= "genmod" then
            error("Cannot write to `KoboldLib.logits` from outside of a generation modifer")
            return
        elseif type(v) ~= "table" then
            error("`KoboldLib.logits` must be a 2D array of numbers; you attempted to set it to a " .. type(v))
            return
        end
        koboldbridge.logits = v
    end


    --==========================================================================
    -- Userscript API: Generated Tokens
    --==========================================================================

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_getters.generated_rows(t)
        local backend = kobold.modelbackend
        if backend == "readonly" or backend == "api" then
            return 0
        elseif koboldbridge.userstate == "outmod" then
            return koboldbridge.num_outputs
        end
        return kobold.settings.numseqs
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_setters.generated_rows(t)
        error("`KoboldLib.generated_rows` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_getters.generated_cols(t)
        if koboldbridge.userstate ~= "genmod" then
            return 0
        end
        local backend = kobold.modelbackend
        if backend == "readonly" or backend == "api" then
            return 0
        end
        return math.tointeger(koboldbridge.generated_cols)
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_setters.generated_cols(t)
        error("`KoboldLib.generated_cols` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return table<integer, table<integer, integer>>
    function KoboldLib_getters.generated(t)
        if koboldbridge.userstate ~= "genmod" and koboldbridge.userstate ~= "outmod" then
            return
        end
        local backend = kobold.modelbackend
        if backend == "readonly" or backend == "api" then
            return
        end
        return koboldbridge.generated
    end

    ---@param t KoboldLib
    ---@param v table<integer, table<integer, integer>>
    function KoboldLib_setters.generated(t, v)
        if koboldbridge.userstate ~= "genmod" then
            error("Cannot write to `KoboldLib.generated` from outside of a generation modifier")
            return
        elseif type(v) ~= "table" then
            error("`KoboldLib.generated` must be a 2D array of integers; you attempted to set it to a " .. type(v))
            return
        end
        koboldbridge.generated = v
    end


    --==========================================================================
    -- Userscript API: Output
    --==========================================================================

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_getters.num_outputs(t)
        local model = kobold.model
        if model == "OAI" or model == "InferKit" then
            return 1
        end
        if koboldbridge.userstate == "outmod" then
            return koboldbridge.num_outputs
        end
        return kobold.settings.numseqs
    end

    ---@param t KoboldLib
    ---@return integer
    function KoboldLib_setters.num_outputs(t)
        error("`KoboldLib.num_outputs` is a read-only attribute")
    end

    ---@param t KoboldLib
    ---@return table<integer, string>
    function KoboldLib_getters.outputs(t)
        if koboldbridge.userstate ~= "outmod" then
            return
        end
        return koboldbridge.outputs
    end

    ---@param t KoboldLib
    ---@param v table<integer, string>
    function KoboldLib_setters.outputs(t, v)
        if koboldbridge.userstate ~= "outmod" then
            error("Cannot write to `KoboldLib.outputs` from outside of an output modifier")
            return
        elseif type(v) ~= "table" then
            error("`KoboldLib.outputs` must be a 1D array of strings; you attempted to set it to a " .. type(v))
            return
        end
        koboldbridge.outputs = v
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
        for i, token in _python.enumerate(bridged.encode(str)) do
            encoded[i+1] = math.tointeger(token)
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
        if type(tok) == "number" then
            tok = {tok}
        end
        local _tok = {}
        local _v
        for k, v in ipairs(tok) do
            _v = math.tointeger(v)
            if _v == nil then
                error "`decode` got a table with one or more non-integer values"
                return
            end
            _tok[k] = _v
        end
        return bridged.decode(_tok)
    end

    ---@return nil
    function kobold.halt_generation()
        koboldbridge.generating = false
    end

    ---@param sequence? integer
    ---@return nil
    function kobold.restart_generation(sequence)
        if sequence == nil then
            sequence = 0
        end
        sequence_type = type(sequence)
        sequence = math.tointeger(sequence)
        if sequence_type ~= "number" then
            error("`kobold.restart_generation` takes an integer greater than or equal to 0 or nil as argument, but got a " .. sequence_type)
            return
        elseif sequence < 0 then
            error("`kobold.restart_generation` takes an integer greater than or equal to 0 or nil as argument, but got `" .. sequence .. "`")
            return
        end
        if koboldbridge.userstate ~= "outmod" then
            error("Can only call `kobold.restart_generation()` from an output modifier")
            return
        end
        koboldbridge.restart_sequence = sequence
    end

    ---@param t KoboldCoreLib
    ---@return string
    function KoboldLib_getters.feedback(t)
        return koboldbridge.feedback
    end

    ---@param t KoboldCoreLib
    ---@param v string
    ---@return KoboldCoreLib
    function KoboldLib_setters.feedback(t, v)
        error("`KoboldLib.feedback` is a read-only attribute")
    end


    --==========================================================================
    -- Core script API
    --==========================================================================

    koboldbridge.userscripts = {}  ---@type table<integer, KoboldUserScriptModule>
    koboldbridge.userscriptmodule_filename_map = {}  ---@type table<KoboldUserScriptModule, string>
    koboldbridge.num_userscripts = 0
    koboldbridge.inmod = nil  ---@type function?
    koboldbridge.genmod = nil  ---@type function?
    koboldbridge.outmod = nil  ---@type function?

    ---@class KoboldUserScript
    ---@field inmod? function
    ---@field genmod? function
    ---@field outmod? function

    ---@class KoboldCoreScript
    ---@field inmod? function
    ---@field genmod? function
    ---@field outmod? function


    ----------------------------------------------------------------------------

    ---@class KoboldUserScriptModule
    ---@field filename string
    ---@field modulename string
    ---@field description string
    ---@field is_config_file_open boolean
    ---@field inmod? function
    ---@field genmod? function
    ---@field outmod? function
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

    ---@param clear? boolean
    ---@return file*
    function KoboldUserScriptModule:get_config_file(clear)
        return get_config_file(koboldbridge.userscriptmodule_filename_map[self], clear)
    end

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
        elseif k == "is_config_file_open" then
            return is_config_file_open(koboldbridge.userscriptmodule_filename_map[t])
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
    ---@return KoboldUserScriptModule?
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
    koboldbridge.logging_name = nil
    koboldbridge.filename = nil

    local sandbox_require_builtins = {
        coroutine = true,
        package = true,
        string = true,
        utf8 = true,
        table = true,
        math = true,
        io = true,
        os = true,
        debug = true,
    }

    local old_load = load
    local function _safe_load(_g)
        return function(chunk, chunkname, mode, env)
            if mode == nil then
                mode = "t"
            elseif mode ~= "t" then
                error("Calling `load` with a `mode` other than 't' is disabled for security reasons")
                return
            end
            if env == nil then
                env = _g
            end
            return old_load(chunk, chunkname, mode, env)
        end
    end

    local old_loadfile = loadfile
    local package_loaded = {}  ---@type table<table, table>
    local old_package_searchers = package.searchers
    ---@param modname string
    ---@param env table<string, any>
    ---@param search_paths? string|table<integer, string>
    ---@return any, string?
    local function requirex(modname, env, search_paths)
        if search_paths == nil then
            search_paths = bridged.lib_paths
        end
        if modname == "bridge" then
            return function() return env.kobold, env.koboldcore end
        end
        if type(modname) == "number" then
            modname = tostring(modname)
        elseif type(modname) ~= "string" then
            error("bad argument #1 to 'require' (string expected, got "..type(modname)..")")
            return
        end
        if sandbox_require_builtins[modname] then
            return env[modname]
        end
        local allowsearch = type(modname) == "string" and string.match(modname, "[^%w._-]") == nil and string.match(modname, "%.%.") == nil
        if allowsearch and package_loaded[env] == nil then
            package_loaded[env] = {}
        elseif allowsearch and package_loaded[env][modname] then
            return package_loaded[env][modname]
        end
        local loader, path
        local errors = {}
        local n_errors = 0
        set_require_path(search_paths)
        for k, v in ipairs(old_package_searchers) do
            loader, path = v(modname)
            if allowsearch and type(loader) == "function" then
                break
            elseif type(loader) == "string" then
                n_errors = n_errors + 1
                errors[n_errors] = "\n\t" .. loader
            end
        end
        set_require_path(bridged.lib_paths)
        if not allowsearch or type(loader) ~= "function" then
            error("module '" .. modname .. "' not found:" .. table.concat(errors))
            return
        end
        local f, err = old_loadfile(path, "t", env)
        if err ~= nil then
            error(err)
            return
        end
        local retval = (f())
        package_loaded[env][modname] = retval == nil or retval
        return package_loaded[env][modname], path
    end
    local function _safe_require(_g)
        ---@param modname string
        ---@return any, string?
        return function(modname)
            return requirex(modname, _g)
        end
    end

    local old_input = io.input
    ---@param file? string|file*
    local function safe_input(file)
        if type(file) == "string" then
            error("Calling `io.input` with a string as argument is disabled for security reasons")
            return
        end
        return old_input(file)
    end

    local old_output = io.output
    ---@param file? string|file*
    local function safe_output(file)
        if type(file) == "string" then
            error("Calling `io.output` with a string as argument is disabled for security reasons")
            return
        end
        return old_output(file)
    end

    local old_lines = io.lines
    ---@param filename? string
    local function safe_lines(filename, ...)
        if type(filename) == "string" then
            error("Calling `io.lines` with a string as first argument is disabled for security reasons")
            return
        end
        return old_lines(filename, ...)
    end

    local function redirected_print(...)
        local args = table.pack(...)
        for i = 1, args.n do
            args[i] = tostring(args[i])
        end
        bridged.print(table.concat(args, "\t"))
    end

    local function _redirected_warn()
        local do_warning = true
        local control_table = {
            ["@on"] = function()
                do_warning = true
            end,
            ["@off"] = function()
                do_warning = false
            end,
        }
        return function(...)
            local args = table.pack(...)
            if args.n == 1 and type(args[1]) == "string" and args[1]:sub(1, 1) == "@" then
                local f = control_table[args[1]]
                if f ~= nil then
                    f()
                end
                return
            end
            if not do_warning then
                return
            end
            for i = 1, args.n do
                args[i] = tostring(args[i])
            end
            bridged.warn(table.concat(args, "\t"))
        end
    end

    local sandbox_template_env = {
        assert = assert,
        connectgarbage = collectgarbage,
        error = error,
        getmetatable = getmetatable,
        ipairs = ipairs,
        load = nil,  ---@type function
        next = next,
        pairs = pairs,
        pcall = pcall,
        print = nil,   ---@type function
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
        warn = nil,   ---@type function
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
        require = nil,  ---@type function
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
            stdin = io.stdin,
            stdout = io.stdout,
            stderr = io.stderr,
            input = safe_input,
            output = safe_output,
            read = io.read,
            write = io.write,
            close = _new_close(io.close),
            lines = safe_lines,
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
            envs[universe].load = _safe_load(env)
            envs[universe].require = _safe_require(env)
            envs[universe].print = redirected_print
            envs[universe].warn = _redirected_warn()
            env._G = env
        end
        return env
    end

    function koboldbridge.obliterate_multiverse()
        for k, v in pairs(config_files) do
            pcall(v.close, v)
        end
        envs = {}
        koboldbridge.userscripts = {}
        koboldbridge.num_userscripts = 0
        koboldbridge.inmod = nil
        koboldbridge.genmod = nil
        koboldbridge.outmod = nil
    end


    --==========================================================================
    -- API for aiserver.py
    --==========================================================================

    ---@return boolean
    function koboldbridge.load_userscripts(filenames, modulenames, descriptions)
        config_files = {}
        config_file_filename_map = {}
        koboldbridge.userscripts = {}
        koboldbridge.userscriptmodule_filename_map = {}
        koboldbridge.num_userscripts = 0
        local has_genmod = false
        for i, filename in _python.enumerate(filenames) do
            bridged.load_callback(filename, modulenames[i])
            koboldbridge.logging_name = modulenames[i]
            koboldbridge.filename = filename
            local f, err = old_loadfile(join_folder_and_filename(bridged.userscript_path, filename), "t", koboldbridge.get_universe(filename))
            if err ~= nil then
                error(err)
                return false
            end
            ---@type KoboldUserScript
            local _userscript = f()
            koboldbridge.logging_name = nil
            koboldbridge.filename = nil
            if _userscript.genmod ~= nil then
                has_genmod = true
            end
            local userscript = deepcopy(KoboldUserScriptModule)
            rawset(userscript, "_inmod", function()
                koboldbridge.logging_name = modulenames[i]
                koboldbridge.filename = filename
                if _userscript.inmod ~= nil then
                    _userscript.inmod()
                end
                koboldbridge:clear_userscript_metadata()
            end)
            rawset(userscript, "_genmod", function()
                koboldbridge.logging_name = modulenames[i]
                koboldbridge.filename = filename
                if _userscript.genmod ~= nil then
                    _userscript.genmod()
                end
                koboldbridge:clear_userscript_metadata()
            end)
            rawset(userscript, "_outmod", function()
                koboldbridge.logging_name = modulenames[i]
                koboldbridge.filename = filename
                if _userscript.outmod ~= nil then
                    _userscript.outmod()
                end
                koboldbridge:clear_userscript_metadata()
            end)
            rawset(userscript, "_filename", filename)
            rawset(userscript, "_modulename", modulenames[i])
            rawset(userscript, "_description", descriptions[i])
            koboldbridge.userscripts[i+1] = userscript
            koboldbridge.userscriptmodule_filename_map[userscript] = filename
            koboldbridge.num_userscripts = i + 1
        end
        return has_genmod
    end

    ---@return nil
    function koboldbridge.load_corescript(filename)
        local f, err = old_loadfile(join_folder_and_filename(bridged.corescript_path, filename), "t", koboldbridge.get_universe(0))
        if err ~= nil then
            error(err)
            return
        end
        ---@type KoboldCoreScript
        local corescript = f()
        koboldbridge.inmod = corescript.inmod
        koboldbridge.genmod = corescript.genmod
        koboldbridge.outmod = corescript.outmod
    end

    function koboldbridge.execute_inmod()
        local r
        koboldbridge:clear_userscript_metadata()
        koboldbridge.restart_sequence = nil
        koboldbridge.userstate = "inmod"
        koboldbridge.regeneration_required = false
        koboldbridge.generating = true
        koboldbridge.generated_cols = 0
        koboldbridge.generated = {}
        if koboldbridge.inmod ~= nil then
            r = koboldbridge.inmod()
        end
        for i = 1, kobold.settings.numseqs do
            koboldbridge.generated[i] = {}
        end
        koboldbridge.outputs = {}
        for i = 1, kobold.num_outputs do
            koboldbridge.outputs[i] = {}
        end
        return r
    end

    ---@return any, boolean
    function koboldbridge.execute_genmod()
        local r
        koboldbridge:clear_userscript_metadata()
        koboldbridge.generating = true
        koboldbridge.userstate = "genmod"
        if koboldbridge.genmod ~= nil then
            local _generated = deepcopy(koboldbridge.generated)
            if not bridged.vars.nogenmod then
                r = koboldbridge.genmod()
            end
            setmetatable(koboldbridge.logits, nil)
            for kr, vr in old_next, koboldbridge.logits, nil do
                setmetatable(vr, nil)
                for kc, vc in old_next, vr, nil do
                    if type(vc) ~= "number" then
                        error("`kobold.logits` must be a 2D table of numbers, but found a non-number element at row " .. kr .. ", column " .. kc)
                        return r
                    end
                end
            end
            setmetatable(koboldbridge.generated, nil)
            for kr, vr in old_next, koboldbridge.generated, nil do
                setmetatable(vr, nil)
                for kc, vc in old_next, vr, nil do
                    if math.tointeger(vc) == nil then
                        error("`kobold.generated` must be a 2D table of integers, but found a non-integer element at row " .. kr .. ", column " .. kc)
                        return r
                    end
                    vr[kc] = math.tointeger(vc)
                    if vr[kc] ~= _generated[kr][kc] then
                        maybe_require_regeneration()
                    end
                end
            end
        end
        koboldbridge.generated_cols = koboldbridge.generated_cols + 1
        return r
    end

    function koboldbridge.execute_outmod()
        local r
        koboldbridge:clear_userscript_metadata()
        koboldbridge.generating = false
        koboldbridge.userstate = "outmod"
        koboldbridge.num_outputs = kobold.settings.numseqs
        if koboldbridge.outmod ~= nil then
            local _outputs = deepcopy(koboldbridge.outputs)
            r = koboldbridge.outmod()
            setmetatable(koboldbridge.outputs, nil)
            for k, v in old_next, koboldbridge.outputs, nil do
                if type(v) ~= "string" then
                    error("`kobold.outputs` must be a 1D array of strings, but found a non-string element at index " .. k)
                    return r
                end
                if v ~= _outputs[k] then
                    maybe_require_regeneration()
                end
            end
        end
        koboldbridge.userstate = nil
        return r
    end


    --==========================================================================
    -- Footer
    --==========================================================================

    metawrapper.__newindex = nil
    setmetatable(KoboldWorldInfoEntry, KoboldWorldInfoEntry_mt)
    setmetatable(KoboldWorldInfoFolder, KoboldWorldInfoFolder_mt)
    setmetatable(KoboldWorldInfoFolderSelector, KoboldWorldInfoFolderSelector_mt)
    setmetatable(KoboldWorldInfo, KoboldWorldInfo_mt)
    setmetatable(KoboldStoryChunk, KoboldStoryChunk_mt)
    setmetatable(KoboldStory, KoboldStory_mt)
    setmetatable(KoboldSettings, KoboldSettings_mt)
    setmetatable(KoboldUserScriptModule, KoboldUserScriptModule_mt)
    setmetatable(KoboldUserScriptList, KoboldUserScriptList_mt)
    setmetatable(kobold, KoboldLib_mt)
    setmetatable(koboldcore, KoboldCoreLib_mt)

    return kobold, koboldcore, koboldbridge
end
