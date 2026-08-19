---
title: Communities of Practice
description: Schaffung fester teamübergreifender Gruppen rund um ein gemeinsames
  Handwerk — Testing, ein Subsystem, eine Sprache — sodass sich Wissen und Standards
  horizontal statt über das Management verbreiten.
category:
- Team
- Communication
- Culture
problems:
- team-silos
- knowledge-silos
- inconsistent-execution
- skill-development-gaps
- limited-team-learning
- duplicated-research-effort
- technology-isolation
- duplicated-effort
- inconsistent-knowledge-acquisition
- undefined-code-style-guidelines
- technology-stack-fragmentation
- slow-knowledge-transfer
- team-confusion
- author-frustration
- automated-tooling-ineffectiveness
- code-duplication
- communication-risk-within-project
- convenience-driven-development
- difficult-to-understand-code
- extended-research-time
- fear-of-conflict
- high-turnover
- inappropriate-skillset
- inconsistent-naming-conventions
- individual-recognition-culture
- inexperienced-developers
- insufficient-design-skills
- language-barriers
- legacy-skill-shortage
- mentor-burnout
- new-hire-frustration
- nitpicking-culture
- procedural-programming-in-oop-languages
- reduced-review-participation
- reduced-team-flexibility
- team-churn-impact
- team-members-not-engaged-in-review-process
- unclear-sharing-expectations
- uneven-workload-distribution
- unmotivated-employees
layout: solution
lang: de
en_slug: communities-of-practice
related_solutions:
- slug: user-communities
  similarity: 0.75
- slug: pair-and-mob-programming
  similarity: 0.7
- slug: internal-technical-coaching
  similarity: 0.7
- slug: knowledge-rotation
  similarity: 0.7
- slug: team-boundaries-aligned-to-architecture
  similarity: 0.7
- slug: code-reading-sessions
  similarity: 0.65
---

## Description

Eine Community of Practice ist eine feste, freiwillige Gruppe von Menschen aus verschiedenen Teams, die ein Handwerk oder ein Anliegen teilen — Testing, die Datenbank, eine Programmiersprache, ein bestimmtes Legacy-Subsystem —, und die sich regelmäßig treffen, um auszutauschen, was sie lernen, und sich auf gemeinsame Praxis zu einigen. Sie bietet einen Kanal, der strukturell in teamgetriebenen Organisationen fehlt. Sobald Teams um Fähigkeiten oder Produkte herum organisiert sind, was für Lieferung üblicherweise korrekt ist, hört Wissen auf, seitwärts zu fließen: Vier Teams lösen dasselbe Testproblem viermal, und vier verschiedene Konventionen entstehen für dieselbe Sache. Eine Community of Practice ist die horizontale Verbindung, die diesen Fluss wiederherstellt, ohne irgendjemanden zu reorganisieren. In Legacy-Landschaften hat sie einen zweiten Nutzen, da die Menschen, die ein gegebenes jahrzehntealtes Subsystem kennen, häufig über Teams verstreut sind, und die Community möglicherweise der einzige Ort ist, wo sie je miteinander sprechen.

## How to Apply ◆

> In einem langlebigen System ist die Expertise zu jedem gegebenen Subsystem üblicherweise durch historischen Zufall über Teams verteilt, und niemand hat sie je in einem Raum versammelt.

- **Formen Sie sich um ein genuines gemeinsames Anliegen**, nicht um eine organisatorische Kategorie. Eine Community für „Backend-Entwickler" hat nichts Spezifisches zu diskutieren; eine für „das Batch-Verarbeitungssubsystem" oder „wie wir Legacy-Code testen" hat sofort eine Agenda.
- Halten Sie die Mitgliedschaft **freiwillig und selbstgewählt**. Von Management verpflichtete Teilnahme produziert Räume voller Menschen, die darauf warten, dass es endet. Eine Community, die niemand besuchen möchte, sagt Ihnen, dass das Thema kein gemeinsames Anliegen ist.
- Geben Sie ihr einen **benannten Koordinator** mit einer kleinen Menge geschützter Zeit. Communities ohne jemanden, der für Terminplanung und Agenda verantwortlich ist, hören still nach der dritten Sitzung auf sich zu treffen, und das Scheitern ist graduell genug, dass es niemand bemerkt.
- Treffen Sie sich in einem **vorhersehbaren Rhythmus** — monatlich funktioniert für die meisten. Wöchentlich ist zu anspruchsvoll für eine freiwillige Gruppe; vierteljährlich zu selten, um die Beziehungen aufzubauen, die den Austausch funktionieren lassen.
- **Verankern Sie Sitzungen in echter Arbeit.** Ein Mitglied geht ein Problem durch, dem es begegnet, eine Lösung, die es gebaut hat, oder einen Vorfall, den es behandelt hat. Präsentationen zu allgemeinen Themen ziehen einmal Publikum an, und dann verfällt die Teilnahme.
- Geben Sie der Community **Autorität über gemeinsame Konventionen** in ihrer Domäne — Coding-Standards für eine Sprache, Testansatz, die Schnittstelle der gemeinsamen Bibliothek. Empfehlungen ohne Autorität werden ignoriert, und die Community wird zu einer Diskussionsgruppe. Dies ist der Unterschied zwischen einer, die zählt, und einer, die nicht zählt.
- Produzieren Sie **etwas Dauerhaftes** aus jeder Sitzung: eine Entscheidung, eine Notiz in einem gemeinsamen Raum, eine Ergänzung zu einer Konvention. Eine Community, die nur Gespräch generiert, verliert ihre Rechtfertigung, sobald jemand die Zeit hinterfragt.
- Nutzen Sie sie bewusst für **Legacy-Subsystemwissen**. Eine Community um ein spezifisches altes System, die alle anzieht, die es berühren, unabhängig vom Team, ist oft der schnellste Weg, ein fragmentiertes Bild davon zu konsolidieren, wie es tatsächlich funktioniert.
- **Lassen Sie Communities enden.** Wenn ein Thema geklärt ist oder sich das gemeinsame Anliegen auflöst, ist das Schließen der Gruppe ein Erfolg statt eines Fehlschlags. Zombie-Communities verbrauchen Kalenderzeit und diskreditieren das Format.

## Tradeoffs ⇄

> Communities stellen horizontalen Wissensfluss günstig wieder her, aber sie verbrauchen Zeit über viele Teams hinweg und verfallen ohne einen Eigentümer und ohne echte Autorität.

**Vorteile:**

- Wissen überschreitet Teamgrenzen, ohne Reorganisation zu erfordern, was genau die Lücke ist, die fähigkeitsgetriebene Teams schaffen.
- Doppelte Untersuchung sinkt, weil jemand im Raum das Problem üblicherweise bereits gelöst hat oder weiß, warum es nicht gelöst werden kann.
- Konventionen konvergieren über Teams durch Übereinstimmung unter Praktikern statt durch architektonisches Dekret, was sie haltbar macht.
- Verstreute Expertise zu einem Legacy-Subsystem wird konsolidiert, und die Community wird oft zum De-facto-Eigentümer von Wissen, das keinen Eigentümer hatte.
- Entwickler erhalten berufliche Entwicklung und eine Peer-Gruppe außerhalb ihres unmittelbaren Teams, was die Bindung unter Spezialisten messbar beeinflusst.

**Kosten und Risiken:**

- Meeting-Zeit häuft sich über viele Menschen an, und die Kosten sind real, während der Nutzen diffus und schwer zuzuschreiben ist.
- Communities verfallen still. Ohne einen Koordinator mit geschützter Zeit hören sie auf sich zu treffen, und niemand trifft eine Entscheidung darüber.
- Ohne Autorität über Konventionen werden sie zu Quatschbuden, was die Zeit genau der Menschen verschwendet, deren Zeit am umkämpftesten ist.
- Sie können sich zu Gatekeeping-Gremien entwickeln, die die Präferenzen ihrer lautesten Mitglieder Teams auferlegen, die nicht vertreten waren.
- In Organisationen unter anhaltendem Lieferdruck ist freiwillige teamübergreifende Aktivität das Erste, was aus Kalendern verschwindet, sodass das Format sichtbare Managementunterstützung braucht, um zu überleben.

## How It Could Be

Eine Organisation mit sechs Produktteams stellte fest, dass jedes unabhängig seinen eigenen Ansatz zum Testen von Legacy-Code gebaut hatte, und dass vier der sechs separat geschlossen hatten, dass es in ihrem Bereich unmöglich sei. Eine Testing-Community-of-Practice wurde gebildet, mit einem Koordinator, dem vier Stunden pro Monat gegeben wurden, und einer monatlichen Sitzung. Die zweite Sitzung war eine Entwicklerin, die Extract-and-Override an einer echten untestbaren Klasse aus der Codebasis ihres Teams demonstrierte. Drei andere Teams wandten die Technik innerhalb eines Monats an. Bis zur sechsten Sitzung hatte die Community einen gemeinsamen Satz von Konventionen für Charakterisierungstests vereinbart und eine kleine gemeinsame Bibliothek von Testfixtures für die zwei am häufigsten gestubbten externen Services gebaut — Arbeit, die kein einzelnes Team allein gerechtfertigt hätte und von der alle sechs nun abhängen.

Eine zweite Community bildete sich um ein spezifisches 1990er-Mainframe-Subsystem, das elf Personen über vier Teams gelegentlich berührten, von denen keiner es vollständig verstand. Die ersten drei Sitzungen wurden damit verbracht, gemeinsam zu rekonstruieren, was es tat, und die Notizen aus diesen Sitzungen wurden zur ersten Dokumentation, die das Subsystem je hatte. Das wertvollste Ergebnis war weniger erwartet: Zwei der elf entdeckten, dass sie beide glaubten, separate Schnittstellen zu pflegen, die sich als zwei Einstiegspunkte in denselben Codepfad mit subtil unterschiedlicher Validierung herausstellten. Einer von ihnen war die Quelle eines intermittierenden Datenproblems gewesen, das das andere Team separat seit vier Monaten untersuchte.
