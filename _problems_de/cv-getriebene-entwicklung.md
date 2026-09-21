---
title: CV-getriebene Entwicklung
description: Technologien oder Praktiken werden primär gewählt, um den eigenen Lebenslauf
  aufzuwerten, statt um Geschäftsprobleme zu lösen.
category:
- Code
- Communication
- Process
related_problems:
- slug: convenience-driven-development
  similarity: 0.65
- slug: cargo-culting
  similarity: 0.6
- slug: increased-technical-shortcuts
  similarity: 0.6
- slug: decision-avoidance
  similarity: 0.6
- slug: premature-technology-introduction
  similarity: 0.55
- slug: assumption-based-development
  similarity: 0.55
solutions:
- boring-technologies
- technical-skills-development
- architecture-decision-records
- technical-spike
- architecture-governance
- code-review-guidelines
- decision-rights-and-escalation
- technology-radar
layout: problem
lang: de
en_slug: cv-driven-development
---

## Description

CV-getriebene Entwicklung entsteht, wenn technische Entscheidungen primär getroffen werden, um beeindruckende Technologien, Frameworks oder Methodiken zum Lebenslauf der Entwickler hinzuzufügen, statt um tatsächliche Geschäftsbedürfnisse oder technische Anforderungen zu adressieren. Dies führt zur Übernahme angesagter, komplexer oder hochmoderner Lösungen, die für den Umfang des Projekts, die Expertise des Teams oder langfristige Wartungsbedürfnisse ungeeignet sein können. Die Praxis stellt individuellen Karrierefortschritt über Projekterfolg und nachhaltige Softwareentwicklung.

## Indicators ⟡

- Technologie-Vorschläge konzentrieren sich stark auf Neuheit oder Angesagtheit statt auf Geschäftswert
- Entwickler betonen bei technischen Diskussionen Lerngelegenheiten über Projektanforderungen
- Häufige Anfragen, die neuesten Versionen von Frameworks oder experimentelle Technologien zu nutzen
- Widerstand gegen die Nutzung bewährter, stabiler, aber "langweiliger" Technologien für angemessene Anwendungsfälle
- Technische Entscheidungen fallen mit Jobsuchphasen oder Leistungsbeurteilungen der Entwickler zusammen
- Teammitglieder erwähnen explizit den Lebenslaufaufbau, wenn sie neue Technologien vorschlagen
- Unverhältnismäßiges Interesse an konferenz- oder blogtauglichen technischen Lösungen
- Präferenz für komplexe Architekturen, wenn einfachere Lösungen ausreichen würden

## Symptoms ▲

- [Verfrühte Technologieeinführung](verfruehte-technologieeinfuehrung.md)
<br/>  Technologien werden aufgrund ihres Lebenslaufwerts statt der Projektpassung übernommen, was unreife oder unpassende Werkzeuge in den Stack einführt.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Unnötig komplexe, durch Lebenslaufaufbau getriebene Technologieentscheidungen schaffen Systeme, die teuer zu warten sind, nachdem der ursprüngliche Entwickler gegangen ist.
- [Wissenslücken](wissensluecken.md)
<br/>  Wenn Technologien wegen ihres Lebenslaufwerts statt der Teamexpertise gewählt werden, fehlt den meisten Teammitgliedern das Wissen, effektiv damit zu arbeiten.
- [Fragmentierung des Technologie-Stacks](fragmentierung-des-technologie-stacks.md)
<br/>  Unterschiedliche Entwickler, die ihre bevorzugten lebenslaufaufwertenden Technologien einführen, schaffen einen fragmentierten Stack mit vielen inkompatiblen Werkzeugen.

## Causes ▼

- [Kultur der individuellen Anerkennung](kultur-der-individuellen-anerkennung.md)
<br/>  Eine Kultur, die individuelle Leistungen über Teamerfolg stellt, motiviert Entwickler dazu, persönlichen Karrierefortschritt bei technischen Entscheidungen zu priorisieren.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Wenn das Management langfristige Wartbarkeit nicht berücksichtigt, stoßen Entwickler auf keinen Widerstand, wenn sie Technologien zu ihrem persönlichen Vorteil wählen.
- [Marktdruck](marktdruck.md)
<br/>  Wettbewerbsdruck am Arbeitsmarkt motiviert Entwickler, ihren Lebenslauf mit angesagten Technologien aufzubauen, um ihre Karriereaussichten zu verbessern.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Entwickler, die Technologien für ihren Lebenslauf gewählt haben, verlassen das Unternehmen, sobald sie genug Erfahrung gesammelt haben, um sie aufzulisten, und nehmen kritisches Wissen mit.

## Detection Methods ○

- Überprüfung der Dokumentation technischer Entscheidungen auf geschäftliche Begründung im Vergleich zu Technologievorteilen
- Beobachtung der Korrelation zwischen Technologie-Vorschlägen einzelner Entwickler und ihren Karrierezielen
- Bewertung, ob Technologieentscheidungen mit Teamexpertise und Projektzeitplänen übereinstimmen
- Nachverfolgung von Wartungsaufwand und Fehlerraten in Bereichen mit neueren vs. etablierten Technologien
- Bewertung, ob die technische Komplexität der tatsächlichen Problemkomplexität entspricht
- Befragung ausscheidender Teammitglieder zu ihrer Motivation für technische Entscheidungen
- Vergleich des Projekt-Technologie-Stacks mit Branchenstandards für ähnliche Projekte
- Beobachtung der Rekrutierungsschwierigkeit für Rollen, die den gewählten Technologie-Stack erfordern
- Analyse, ob die Technologieübernahme Branchen-Hype-Zyklen folgt statt Projektbedürfnissen
- Überprüfung von Retrospektiven auf Erwähnungen technologiebezogener Herausforderungen oder Bedauern

## Examples

Ein Senior-Entwickler in einem kleinen E-Commerce-Projekt besteht darauf, das Backend mit einer hochmodernen funktionalen Programmiersprache, einer komplexen Event-Sourcing-Architektur und der neuesten NoSQL-Datenbank umzusetzen, obwohl das Team keine vorherige Erfahrung mit diesen Technologien hat. Die Projektanforderungen sind unkompliziert: Nutzerauthentifizierung, Produktkatalog und Auftragsverarbeitung. Auf Nachfrage erwähnt der Entwickler, dass er auf Konferenzen über diese Technologien sprechen und sein LinkedIn-Profil aktualisieren möchte. Das Ergebnis ist eine sechsmonatige Lieferverzögerung, erhebliche Kostenüberschreitungen und ein System, das nur eine Person warten kann. Nachdem der Entwickler für eine neue Rolle geht, die seine "moderne Architekturerfahrung" hervorhebt, kämpft das verbleibende Team damit, Fehler zu beheben oder Features hinzuzufügen, was letztlich eine komplette Neuimplementierung mit konventionelleren Technologien erfordert.
