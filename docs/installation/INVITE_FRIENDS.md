# Inviting friends

A friend needs an account to make videos with your Nivra. Accounts have a **role**:

| Role | Can do |
|------|--------|
| `member` | make videos in the Nivra App, see, download, stop and delete **only their own** videos, create their own access keys. Cannot see other people's prompts or videos, the queue, workers, models, datasets or training. Limited in video length and number of videos. |
| `viewer` | look at the dashboard, change nothing |
| `operator` | everything about training, datasets, models, jobs and workers; not limited in video length |
| `admin` | also users, projects, project codes, limits and cleanup |

Use `member` for friends.

## Invite someone

**Studio → People → Invite a friend** (or `nivra admin invite --note "Sam" [--role member] [--expires-days 7]` on the
server). You get a link like `https://studio.example.com/join?code=nvi_...`. Send it to your friend.

* The link works **once** and expires (7 days by default, 30 at most). Only a hash of the code is stored.
* Your friend opens it, chooses a username and a password (12 characters or more) and is signed in. You never see or
  send a password.
* If the username is taken or the password is too short the page tells them and the link keeps working.

## What your friend does next

1. Install the Nivra App: send them `Nivra-Setup.bat` (or the release zip).
2. In the App choose **Sign in** and enter the Nivra address (the part of the link before `/join`), their username
   and their password. The App creates an access key for itself and stores only the key, never the password. (Or: in
   Studio → Settings create an access key and paste it into the App.)
3. Type a prompt. The video is made by the computers that help the network.

## Limits

**Studio → Settings → Limits and cleanup** (administrators):

| Limit | Default | Applies to |
|-------|---------|-----------|
| Longest video for members | 10 minutes | members only (operators and admins are exempt) |
| Long videos at the same time per person | 2 | everyone |
| Unfinished videos per person | 50 | everyone |
| Delete temporary files of finished long videos after | 24 hours | the whole project |

Nobody can ask for more than 30 minutes per video for now. Helper computers are shared fairly: when several people
have work waiting, each free computer goes to the person with the fewest running tasks, so one long video cannot
starve everyone else. Training keeps a reserved share of the computers (at least one task in four when it qualifies).

## Take access away

**Studio → People → Turn off**. The account is signed out at once and all its access keys stop working. Turning it
on again restores access. A pending invitation can be withdrawn in the same page.

## Lending a computer

A friend can also let their computer help: in the App, **This computer as a worker**, with the project code from
Studio (top right). It only gets work that fits its memory and graphics card.
