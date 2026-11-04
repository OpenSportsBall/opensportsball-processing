# Security

**Please don't open a public issue.** Report vulnerabilities privately
through GitHub:
[Report a vulnerability](https://github.com/OpenSportsBall/opensportsball-processing/security/advisories/new)
(the repository's **Security** tab → **Report a vulnerability**). Only the
maintainers see it.

This repository processes game footage of children, so issues that could
expose footage, or let a crafted recording or manifest run code or read
other files, are especially important. Include what you found, how to
reproduce it, and what an attacker could do with it. We'll acknowledge
the report within a few days and credit you in the advisory unless you'd
rather we didn't.

The phone app, website, API and AWS infrastructure live in the main
[opensportsball](https://github.com/OpenSportsBall/opensportsball)
repository; see its `SECURITY.md` for that scope and for the rules on
testing against the live service.

Only the latest code on `main` gets security fixes.
