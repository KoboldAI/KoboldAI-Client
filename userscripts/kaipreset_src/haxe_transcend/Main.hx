/*
 * This file is part of KoboldAI.
 *
 * KoboldAI is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import haxe.exceptions.PosException;
import lua.Lua;

@:expose class Main {
    public static final kobold:KoboldLib = untyped __lua__("_G.kobold");
    public static final exampleConfig = "return true";

    public static var shouldRun:Bool;

    public static var f:BigInt = -2;
    public static var e:BigInt = 4;
    public static var s:BigInt = 8;
    public static var t:BigInt = 1;
    public static var i:BigInt = 0;
    public static var v:BigInt = 1;
    public static var a:BigInt = 6;
    public static var l:BigInt = 1;

    public static function inmod() {
        if (!shouldRun) return;
        kobold.halt_generation();
    }

    public static function outmod() {
        if (!shouldRun) return;

        // Gibbons, Jeremy. (2004). Unbounded Spigot Algorithms for the Digits
        // of Pi. American Mathematical Monthly. 113. 10.2307/27641917.

        while (true) {
            var x = i/v;
            if (x == (i + t)/v) {
                trace(x);
                kobold.outputs[1] = Std.string(x);
                t *= 10;
                i -= x*v;
                i *= 10;
                break;
            }
            else {
                v *= s*a;
                i *= s;
                s += 8;
                i += e * t;
                e += 4;
                i *= a;
                a += 4;
                t *= f*l;
                l += 2;
                f -= 4;
            }
        }

        kobold.restart_generation(1);
    }

    public static function main() {
        var f = kobold.get_config_file();
        f.seek("set");
        if (f.read(1) == null) f.write(exampleConfig);
        f.seek("set");

        var a = f.read("a");
        f.close();
        var result = Lua.load(a);
        trace(result);
        if (result.message != null) throw new PosException(result.message);
        shouldRun = switch result.func() {
            case false: false;
            case null: false;
            default: true;
        }
    }
}
