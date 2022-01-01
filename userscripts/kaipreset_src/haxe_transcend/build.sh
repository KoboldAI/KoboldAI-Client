#!/bin/bash
# This file is part of KoboldAI.
#
# KoboldAI is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
set -e
path=$1
if [ $# -eq 0 ]
then
    path="out.lua"
fi
haxe --lua $path -L littleBigInt --main Main
perl -pe 's/\Q_G.require("rex_pcre")/\E({flags = function() return {CASELESS = 1, DOTALL = 1, MULTILINE = 1, UCP = 1, UTF8 = 1} end, gsub = function() return "" end, new = function() return {} end})/' -i $path
perl -pe 's/\Qreturn _hx_exports/\Ereturn _hx_exports.Main/' -i $path
perl -pe "s/\Q  _hx_bit_raw = _G.require('bit32')/\E  _hx_bit_raw = {arshift = function(x, n) local y = x >> n; if (x < 0) then y = y | ~(-1 >> n) end return y end, band = function(x, y) return x & y end, bor = function(x, y) return x | y end, bnot = function(x) return ~x end, bxor = function(x, y) return x ~ y end, lshift = function(x, n) return x << n end, rshift = function(x, n) return x >> n end}/" -i $path
perl -pe 's/\Q__lua_lib_luautf8_Utf8 = _G.require("lua-utf8")/\E__lua_lib_luautf8_Utf8 = {byte = _G.string.byte, find = _G.string.find, gmatch = _G.string.gmatch, gsub = _G.string.gsub, lower = _G.string.lower, match = _G.string.match, reverse = _G.string.reverse, sub = _G.string.sub, upper = _G.string.upper}; for k, v in pairs(_G.utf8) do __lua_lib_luautf8_Utf8[k] = v end;/' -i $path
perl -pe 's/\Q_G.xpcall(Main.main, _hx_error)/\Elocal err; if not xpcall(Main.main, function(obj) err = ""; local _print = _G.print; _G.print = function(...) local args = table.pack(...) for i = 1, args.n do args[i] = tostring(args[i]) end err = err .. table.concat(args, "\\t") .. "\\n" end _hx_error(obj); _G.print = _print end) then _G.error(err) return end;/' -i $path
