# Strava Metro application (Academic Researchers program)

Romit applies on behalf of the team. The program accepts one application per person and one project per application.

**Deadline: Friday 23 October 2026, 5 p.m. Pacific (8 p.m. Boston).** Check this on the application page when you open it. Decisions come out in November.

## What we'd get, and what it costs us

**The data.** Published studies that use Strava Metro describe walking and running counts per street segment on an OpenStreetMap-based network, by hour. Small counts are masked for privacy: 3 or fewer people in an hour shows as 0, and higher counts are rounded up to a multiple of 5. Quiet paths will mostly read as zero.

**The area is Boston only.** The form says the area of interest must match the institution's location, so we can't get Melbourne or New York. Strava's use for us is as rough local counts for the NEU Boston demonstration, which has no public pedestrian counts today. It is validation for the demo, and no Tier 1 deliverable depends on it.

**The obligations** (from the agreement at the end of the form):
- A representative of each supporting organisation must attend an onboarding meeting, and we must attend at least one check-in each quarter for the year.
- We share the final project with Strava, and Strava may use our names and Northeastern's in case studies and other public material.
- Access lasts one year.
- Everyone who works with the data needs their own Strava Metro login.

**How we may use it.** Treat it like other academic-terms data in this project (`CLAUDE.md`, core rule 5): validation and research only. No model trained on Strava data goes into anything we ship or distribute, and we don't redistribute the data. If we're accepted, add it to `docs/data_licenses.md` and write a data card before anyone downloads it.

## Before submitting

1. **Get a letter of support. This takes the longest, so start now.** The form treats an application without at least one letter as incomplete. The letter must come from a government agency, nonprofit or community organisation, and all letters go in as one PDF. Options, roughly in order of fit:
   - **WalkBoston** (Boston's pedestrian advocacy nonprofit). Walking is their whole remit.
   - **Boston Transportation Department** (City of Boston). A strong letter, but likely slower to get.
   - **Northeastern's Campus Planning office.** Easy to reach and the demo is their campus. It's part of our own institution, though, and the form asks for partner organisations, so use it alongside an outside letter rather than instead of one.

   Use the email template below. Tell them about the onboarding meeting and quarterly check-ins up front, since their representative has to attend.
2. **Agree as a team:**
   - Who gets a Metro login (only people who will touch the data).
   - Whether the team is fine with Strava using our names publicly.
   - Whether anyone can keep the check-ins going after the course ends in December (access runs a full year).
   - Whether the instructor should be named as faculty contact.
3. **Fill in the placeholders** in the answers below: roles, the supporting organisation and its contact.
4. **Submit by Thursday 22 October**, a day early.

## Form answers

Each answer is under the 500-character limit; the count is in brackets. Recount after filling in placeholders.

### First page

- **Project type:** Transportation planning. Leave Trail use and Tourism unselected.
- **Job title:** Graduate Student Researcher
- **Project description** (100–500 characters) [397]:

  > Stoa is a Northeastern University graduate course project that predicts hourly pedestrian volumes on streets and paths, so planners can test changes such as path closures or event days before making them. We would use Strava Metro walking and running counts for Boston to check our model's route-choice patterns. Use is research and evaluation only: no individual-level data and no commercial use.

### Application

**Project title** [83]

> Stoa: predicting and checking pedestrian volumes on Boston streets and campus paths

**Describe your project in detail** [491]

> Stoa is an open-source tool that predicts hourly pedestrian volumes on every street and path of a site, so planners can test changes (closing a path, adding an entrance, an event day) before making them. The model learns from automated counts in Melbourne, Australia, and will be tested on NYC DOT counts. Our demonstration site, the area around Northeastern's Boston campus, has no public pedestrian counts. Strava Metro walking and running counts would let us check our Boston predictions.

**Project goal** [425]

> To measure how well our predicted pedestrian volumes for Boston match Strava Metro walking and running counts, segment by segment and by hour, and to map where they differ. This tells planners how far a model trained in another city can be trusted in Boston. It also shows where Strava's sample, which leans toward runners and recreational walks, needs correcting before it can stand in for counts on everyday walking routes.

**What preparation have you done for this project to date?** [450]

> We built an automated data pipeline (Airflow, versioned data) and a walking network with street-level features from OpenStreetMap: amenities and opening hours, transit access, building cover and street layout. It runs on Melbourne today, and the same code builds Boston, where we already hold terrain, building heights, aerial imagery and hourly weather. The model is being trained on about 100 Melbourne sensors, to be tested on 114 NYC count sites.

**What is your plan to publish your project's findings?** [376]

> Code and methods go in our public GitHub repository. Results go in our course final report and our presentation at Northeastern's project expo in December 2026, plus a written comparison of Strava counts with our predictions, shared with Strava and our supporting organization. If the results hold up, we will submit a short paper to a transportation or urban analytics venue.

**What is your plan to create awareness and promote your project?** [327]

> We will present at Northeastern's course expo in December 2026, publish the repository and write-up openly, and share the results with [supporting organization] and other Boston groups working on walking. Team members will post about the project, crediting Strava Metro, and we will give Strava a short case summary it can use.

**How will you measure the success of your project?** [374]

> 1) Rank correlation between our predicted and Strava's walking volumes across Boston street segments: does the model find the busy and quiet streets? 2) Agreement by hour of day and on weekdays vs weekends. 3) A map of where the two disagree, with likely reasons, such as popular running routes. 4) A public, reproducible comparison that others can re-run on their own city.

**What are the potential barriers to success and alternative plans to ensure project success?** [419]

> Privacy masking zeroes or rounds small counts, so quiet paths may read as zero; we will also compare at daily and monthly totals and over larger areas. Strava over-represents runners and recreation, so we compare rankings and relative patterns, not raw totals. If coverage near campus is thin, we widen the area to nearby neighborhoods. Our core model work does not depend on Strava, so a delay won't stall the project.

**Can you complete your project in that timeframe?** Yes.

**Do you have the support of your local government agencies, nonprofits, and/or other community organizations?** Yes, once the letter is in hand. Don't submit before then.

**If yes, list those organizations and their contacts** [175 with placeholders]

> [Organization], [contact name], [title], [email]. [One line on their role, e.g. "WalkBoston will review our Boston results and attend the onboarding and quarterly check-ins."]

**List everyone working on this project from your organization and their roles** [285 with placeholders]

> Romit Basak (applicant; data and feature pipeline lead), Puneet Singh Puri (pipeline orchestration), Arshmehar Kaur ([role]), Bimbasri Paduru ([role]) and Shubham Madhusudan Hadawle ([role]). All are graduate students at Northeastern University in IE7374 (Machine Learning Operations).

Add "Faculty contact: [name], [title]" at the end if the instructor agrees.

**Why should yours be one of those selected?** [440]

> Much of the published work with Strava Metro studies cycling; we study walking, where cities have far fewer counts. Stoa is open source and built for planners, so the comparison has direct use: it shows when a model trained in one city can be trusted in another, and how to correct Strava's sample for everyday walking. Our pipeline and Boston street network are already built on OpenStreetMap, so we can start as soon as access is granted.

**What is your role with the research institution?** Graduate student.

**Agreement checkboxes:** read them with the team first (see "The obligations" above).

## Email asking for a letter of support

> **Subject:** Letter of support for a Northeastern student project on Boston pedestrian volumes (by 21 October)
>
> Dear [name],
>
> I'm a graduate student at Northeastern University. My team is building Stoa, an open-source tool that predicts hourly pedestrian volumes on every street and path of an area, so planners can test changes such as closing a path or adding an entrance before making them. Our demonstration site is the area around Northeastern's Boston campus.
>
> We're applying to Strava Metro's Academic Researchers program, which would give us a year of Strava walking and running counts for Boston to check our predictions. The application needs a letter of support from a local organisation, and [organisation] is the natural fit because [reason, e.g. "of your work on walkability in Boston"].
>
> The letter can be short: that you know of the project and support it, and how the results could be useful to you. If we're selected, Strava asks a representative of each supporting organisation to join an onboarding call and a check-in once a quarter for the year. We'd share all results with you.
>
> The deadline is 23 October, so a letter by 21 October would give us time to submit. I've attached a draft you're welcome to edit or replace, and I'm happy to talk before then.
>
> Thank you,
> Romit Basak
> Graduate Student, Northeastern University
> basak.r@northeastern.edu

## Draft letter (for the organisation to edit and sign on its letterhead)

> [Date]
>
> Strava Metro, Academic Researchers Program
>
> To the selection committee:
>
> [Organisation] supports the application by Romit Basak and team at Northeastern University to the Strava Metro for Academic Researchers program for their project, Stoa.
>
> Stoa predicts hourly pedestrian volumes on Boston streets and paths and lets planners test changes before making them. Boston has few public pedestrian counts, so comparing these predictions with Strava Metro walking and running data would show where model estimates can be relied on and where they can't. [One or two sentences on why this matters to the organisation, in its own words.]
>
> If the application is selected, [name, title] will represent [organisation] at the onboarding meeting and the quarterly check-ins, and we look forward to reviewing the team's results.
>
> Sincerely,
> [Name], [Title]
> [Organisation], [contact details]

## If we're accepted

- Add Strava Metro to `docs/data_licenses.md`: academic programme terms, validation and research only, no redistribution, one year.
- Write `docs/data_cards/strava_metro.md` (coverage, masking rules, segment IDs and how they map to our OSM graph) before downloading anything.
- Raw Strava files stay in `data/raw/strava_metro/`: never committed to git or copied outside the team's own storage.
- Calendar: onboarding meeting, then quarterly check-ins through autumn 2027.
