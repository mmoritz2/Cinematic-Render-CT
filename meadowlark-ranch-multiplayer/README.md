# Meadowlark Ranch — enable multiplayer

This agent is attached to **Cinematic-Render-CT** and cannot push to
[`mmoritz2/meadowlark-ranch`](https://github.com/mmoritz2/meadowlark-ranch),
which is the repo behind https://mmoritz2.github.io/meadowlark-ranch/.

The game already had a full club/chat/rider-sync layer over MQTT. It shipped
switched off. This patch turns it on.

## What players get

- Opening the game joins the public **kestrel** club so other riders appear
- **🌐 Club** to pick a name, copy an invite link (`?club=CODE`), or use a private code
- Chat, speech bubbles, club boards, and club breeding stay as they were
- Other riders show on the mini-map and world map (teal dots)
- Jump / fly height is synced so remotes leave the ground
- `?solo=1` stays fully offline and skips the MQTT download

## Apply it to the live game

```bash
git clone https://github.com/mmoritz2/meadowlark-ranch.git
cd meadowlark-ranch
git checkout -b cursor/enable-multiplayer-bd33
git apply /path/to/0001-enable-multiplayer.patch
git commit -am "Enable multiplayer clubs by default"
git push -u origin cursor/enable-multiplayer-bd33
```

Then open a PR on `mmoritz2/meadowlark-ranch` and merge it. GitHub Pages will
pick up `ranch3d.html` from `main`.

To try it locally first:

```bash
python -m http.server 8431
```

Open two browser windows to http://127.0.0.1:8431/ranch3d.html — after a few
seconds both should join club `kestrel` and see each other.
