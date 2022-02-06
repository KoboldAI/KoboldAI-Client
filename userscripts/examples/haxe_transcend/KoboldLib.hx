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

import lua.Table;

extern class KoboldLib {
    @:luaDotMethod public function get_config_file(?clear:Bool):lua.FileHandle;
    @:luaDotMethod public function halt_generation():Void;
    @:luaDotMethod public function restart_generation(?sequence:Int):Void;
    public var outputs:Null<Table<Int, String>>;
}
