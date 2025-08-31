using System.Reflection;
using Microsoft.Xna.Framework;
using Netcode;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Objects;
using StardewModdingAPI.Events;
using StardewValley.Tools;
using StardewValley.Locations;
using StardewValley.Menus;
using static StardewValley.Menus.LoadGameMenu;
using StardewValley.TerrainFeatures;

namespace testUtils
{
	public static class TestUtils
	{
        static String TEST_UTILS_LOG = "Test Utils Log:";
        public static readonly Dictionary<string, int> ItemIdMap = new Dictionary<string, int>
    {
        { "copperBar", 334 },
        { "ironBar", 335 },
        { "goldBar", 336 },
        { "iridiumBar", 337 },
        { "wood", 388 },
        { "stone", 390 },
        { "copper", 378 },
        { "iron", 380 },
        { "coal", 382 },
        { "gold", 384 },
        { "iridium", 386 },
        { "stardrop", 434 },
    };
        public static readonly Dictionary<string, int> ToolIdMap = new Dictionary<string, int>
    {
        { "Axw", 0 },
        { "How", 1 },
        { "FishingRod", 2 },
        { "Pickaxe", 3 },
        { "WateringCan", 4 },
        { "MeleeWeapon", 5 },
        { "Slingshot", 6 },
    };
        public static void add_chest(int posx, int posy, Color color, Mod mod)
		{
            Vector2 chestPosition = new Vector2(posx, posy - 1);
        
            var newChest = new Chest(true);
            var brownTint = new NetColor(value: Color.Brown);
            Type chestType = typeof(Chest);
            FieldInfo? tintField = chestType.GetField("tint", BindingFlags.NonPublic | BindingFlags.Instance);

       
            if (tintField != null)
            {
               
                try
                {
                    
                    tintField.SetValue(newChest, Color.Brown); 
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to set the 'tint' field value: {ex.Message}");
                }
            }

          
            Game1.getFarm().objects.Add(chestPosition, newChest);
            mod.Monitor.Log($"{TEST_UTILS_LOG} Chest is added at ({posx}, {posy}). time: {System.DateTime.Now}");
        }

        public static bool callTryToPurchaseItem(object shopMenuInstance, ISalable item, ISalable? held_item, int stockToBuy)
        {
           
            Type shopMenuType = shopMenuInstance.GetType();

            
            MethodInfo? tryToPurchaseItemMethod = shopMenuType.GetMethod("tryToPurchaseItem", BindingFlags.NonPublic | BindingFlags.Instance);

            if (tryToPurchaseItemMethod != null)
            {
              
                var ret = tryToPurchaseItemMethod.Invoke(shopMenuInstance, new object[] { item, held_item, stockToBuy, 0, 0 });
                return ret!=null && (bool)ret == true;
            }
            else
            {
                Console.WriteLine("method tryToPurchaseItem unfind!");
                return false;
            }
        }

        public static void print_mouse_pos(Mod mod)
        {
            mod.Monitor.Log($"OldMouseX: {Game1.getOldMouseX(ui_scale: false)}, OldMouseY: {Game1.getOldMouseY(ui_scale: false)}");
            mod.Monitor.Log($"OldMouseX2: {Game1.oldMouseState.X}, OldMouseY: {Game1.oldMouseState.Y}");

        }

        public static void give_thing(Tool tool, Mod mod)
        {

            
            var playerInventory = Game1.player.Items;

            
            for (int i = 0; i < playerInventory.Count; i++)
            {
                if (playerInventory[i] == null)
                {
                    playerInventory[i] = tool;
                    mod.Monitor.Log($"{tool} added to player's inventory.", LogLevel.Info);
                    return;
                }
            }

            
            mod.Monitor.Log($"Player's inventory is full. Could not add {tool}.", LogLevel.Warn);
        }

        public static void give_tool(string toolName, Mod mod)
        {
            try
            {
             
                var playerInventory = Game1.player.Items;
                Tool tool = getToolInstance(ToolIdMap[toolName]);


                
                for (int i = 0; i < playerInventory.Count; i++)
                {
                    if (playerInventory[i] == null)
                    {
                        playerInventory[i] = tool;
                        mod.Monitor.Log($"{tool} added to player's inventory.", LogLevel.Info);
                        return;
                    }
                }

                
                mod.Monitor.Log($"Player's inventory is full. Could not add {tool}.", LogLevel.Warn);
            }
            catch (Exception ex)
            {
                mod.Monitor.Log($"{ex.Data}");
            }
            
        }


        private static Tool getToolInstance(int id)
        {
        switch (id)
        {
            case 0:
                return new Axe();
            case 1:
                return new Hoe();
            case 2:
                return new FishingRod();
            case 3:
                return new Pickaxe();
            case 4:
                return new WateringCan();
            case 5:
                return new MeleeWeapon();
            case 6:
                return new Slingshot();
            default:
                throw new ArgumentException($"invalid tool ID: {id}");
        }
    }


        public static void remove_items(Item item)
        {
            var player = Game1.player;
            player.removeItemFromInventory(item);
        }

        public static void set_time(Mod mod)
        {
            Game1.timeOfDay = 1200;
            
        }

        public static void tp_player(string location, Mod mod)
        {

            // on the map
            if (Game1.currentLocation != null)
            {
                
                var currentLocation = Game1.currentLocation;

               
                if (Game1.getLocationFromName(location) is Beach)
                {

                   
                    Game1.warpFarmer("Beach", 10, 10, false);

                    mod.Monitor.Log("Player teleported to the beach.", LogLevel.Info);
                }
                else
                {
                    var gameLocation = Game1.getLocationFromName(location);
                    Game1.warpFarmer(location, 20, 20, false);
                    gameLocation.resetForPlayerEntry();
                    mod.Monitor.Log("location not found.", LogLevel.Warn);
                }
            }
            else
            {
                mod.Monitor.Log("Player has no location. Cannot teleport.", LogLevel.Warn);
            }
        }

        public static NPC? GetNearestNPC(Farmer player)
        {
            
            NPC? nearestNpc = null;
            double nearestDistance = double.MaxValue;
            Vector2 standingPosition = player.getStandingPosition();
            var playerTileLocation = new Vector2(
                standingPosition.X / Game1.tileSize,
                standingPosition.Y / Game1.tileSize
            );
            

            foreach (NPC npc in Game1.currentLocation.characters)
            {
                Vector2 npcStandingPosition = npc.getStandingPosition();
                var npcTileLocation = new Vector2(
                    npcStandingPosition.X / Game1.tileSize,
                    npcStandingPosition.Y / Game1.tileSize
                );
               
                double distance = Vector2.Distance(playerTileLocation, npcTileLocation);

                
                if (distance < nearestDistance)
                {
                    nearestDistance = distance;
                    nearestNpc = npc;
                }
            }

            return nearestNpc;
        }

        public static void enterLoadGameMenu(Mod mod, Action onComplete)
        {
            var bufferFrames = 5;
            EventHandler<UpdateTickedEventArgs>? checkEnterLoadGameMenu = null;
            checkEnterLoadGameMenu = (object? sender, UpdateTickedEventArgs e) =>
            {
                IClickableMenu activeClickableMenu = Game1.activeClickableMenu;
                if (activeClickableMenu is TitleMenu titleMenu)
                {
                    if (bufferFrames <= 0)
                    {
                        mod.Helper.Events.GameLoop.UpdateTicked -= checkEnterLoadGameMenu;
                        titleMenu.performButtonAction("Load");
                        onComplete();
                    }
                    else
                    {
                        bufferFrames--;
                    }
                }
            };
            mod.Helper.Events.GameLoop.UpdateTicked += checkEnterLoadGameMenu;
        }

        public static void loadGame(string which, Mod mod, Action onComplete)
        {
            var bufferFrames = 5;
            EventHandler<UpdateTickedEventArgs>? checkLoadGame = null;
            checkLoadGame = (object? sender, UpdateTickedEventArgs e) =>
            {
                IClickableMenu activeClickableMenu = Game1.activeClickableMenu;
                if (activeClickableMenu is TitleMenu titleMenu)
                {
                    if (TitleMenu.subMenu is LoadGameMenu loadGameMenu)
                    {
                        if (bufferFrames <= 0)
                        {
                            mod.Helper.Events.GameLoop.UpdateTicked -= checkLoadGame;
                            int index = int.Parse(which);
                            var slot = loadGameMenu.MenuSlots[index];
                            if (slot is SaveFileSlot saveFileSlot)
                            {
                                saveFileSlot.Activate();
                            }
                            onComplete();
                        }
                        else
                        {
                            bufferFrames--;
                        }
                    }
                    
                }
            };
            mod.Helper.Events.GameLoop.UpdateTicked += checkLoadGame;
        }


        public static void exitGameToTitle()
        {
            Game1.ExitToTitle();
        }

        public static void give_items(string item_id, int amount)
        {
           
            var item = new StardewValley.Object(item_id, amount);
            Game1.player.addItemToInventoryBool(item);
        }

        public static void give_money(int amount)
        {
            Game1.player.Money += amount;
        }

        public static Vector2 getTileLocation(Farmer player)
        {
            Vector2 standingPosition = player.getStandingPosition();
            var playerTileLocation = new Vector2(
                standingPosition.X / Game1.tileSize,
                standingPosition.Y / Game1.tileSize
            );
            return playerTileLocation;
        }

    }

    
}

