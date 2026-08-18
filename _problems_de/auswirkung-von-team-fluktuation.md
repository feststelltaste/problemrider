---
title: Auswirkung von Team-Fluktuation
description: Über die Zeit bringen Entwickler, die dem Team beitreten und es verlassen,
  inkonsistente Praktiken und Wissenslücken mit, die die Codequalität verschlechtern.
category:
- Code
- Communication
- Process
related_problems:
- slug: high-turnover
  similarity: 0.7
- slug: lower-code-quality
  similarity: 0.65
- slug: difficult-developer-onboarding
  similarity: 0.6
- slug: knowledge-gaps
  similarity: 0.6
- slug: knowledge-dependency
  similarity: 0.6
- slug: change-management-chaos
  similarity: 0.6
solutions:
- structured-onboarding-program
- knowledge-rotation
- documentation-as-code
- pair-and-mob-programming
- knowledge-base
- living-documentation
- architecture-decision-records
- code-reading-sessions
- communities-of-practice
- written-first-communication
layout: problem
lang: de
en_slug: team-churn-impact
---

## Description

Die Auswirkung von Team-Fluktuation bezieht sich auf die negativen Effekte auf Codequalität, Konsistenz und Systemwissen, die aus häufigen Änderungen in der Teamzusammensetzung resultieren. Während Entwickler gehen, nehmen sie wertvolles Systemwissen mit, während neue Teammitglieder unterschiedliche Coding-Stile, Praktiken und Annahmen mitbringen. Ohne starke Prozesse zur Verwaltung dieses Übergangs wird die Codebasis graduell inkonsistent, undokumentierte Entscheidungen werden vergessen, und das Gesamtsystem wird schwieriger zu warten.

## Indicators ⟡
- Erhebliche Unterschiede in Codestil und Ansatz zwischen verschiedenen Teilen des Systems
- Kritisches Systemwissen existiert nur in den Köpfen bestimmter Personen
- Neue Teammitglieder brauchen länger als erwartet, um produktiv zu werden
- Code-Review-Diskussionen beinhalten häufig Debatten über historische Designentscheidungen
- Dokumentationslücken in Bereichen, in denen wichtige Mitwirkende gegangen sind

## Symptoms ▲

- [Inkonsistente Codebasis](inkonsistente-codebasis.md)
<br/>  Während Entwickler durch das Team zirkulieren und unterschiedliche Coding-Stile und Praktiken mitbringen, wird die Codebasis in Mustern, Konventionen und Ansätzen inkonsistent.
- [Wissenssilos](wissenssilos.md)
<br/>  Wenn erfahrene Entwickler gehen, konzentriert sich kritisches Systemwissen auf weniger Personen, was gefährliche Wissenssilos schafft.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Neue Teammitglieder, die mit bestehenden Konventionen und Designentscheidungen nicht vertraut sind, führen Code ein, der nicht etablierten Mustern folgt, was die Gesamtqualität verschlechtert.
- [Verringerte Teamproduktivität](verringerte-teamproduktivitaet.md)
<br/>  Lange Onboarding-Zeiten und verlorenes institutionelles Wissen verlangsamen das Team, während neue Mitglieder darum kämpfen, produktiv zu werden.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Neue Entwickler, die ursprüngliche Designentscheidungen nicht verstehen, erstellen Workarounds statt ordentlicher Lösungen, was Komplexität hinzufügt.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Wenn Entwickler gehen, ohne ihr Wissen zu dokumentieren, geht kritische Systeminformation verloren, was anhaltende Dokumentationslücken schafft.

## Causes ▼

- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Häufiges Verlassen des Teams durch Mitglieder ist der direkte Treiber der Team-Fluktuation und des daraus resultierenden Wissensverlusts und der Inkonsistenz.
- [Unzureichendes Onboarding](unzureichendes-onboarding.md)
<br/>  Ohne effektive Onboarding-Prozesse übernehmen neue Teammitglieder ihre eigenen Praktiken statt bestehende Konventionen zu lernen, was die Auswirkung der Fluktuation verstärkt.
- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne dokumentierte Coding-Standards und architektonische Richtlinien bringt jeder neue Entwickler seinen eigenen Ansatz mit, was Fluktuation schädlicher macht.

## Detection Methods ○
- **Fluktuationsraten-Analyse:** Verfolgung der Häufigkeit von Abgängen von Teammitgliedern und ihrer Auswirkungsdauer
- **Wissens-Audit:** Identifikation kritischen Wissens, das nur bei bestimmten Personen existiert
- **Code-Konsistenzanalyse:** Nutzung von Werkzeugen zur Messung der Stil- und Musterkonsistenz über die Codebasis hinweg
- **Onboarding-Zeit-Metriken:** Verfolgung, wie lange neue Teammitglieder brauchen, um produktiv zu werden
- **Dokumentationsabdeckung:** Bewertung, welches kritische Systemwissen ordentlich dokumentiert ist

## Examples

Ein Zahlungsverarbeitungssystem wurde ursprünglich von einem eng zusammenarbeitenden Team gebaut, das konstant kommunizierte und tiefes Verständnis der Geschäftsanforderungen teilte. Über drei Jahre verließen alle ursprünglichen Teammitglieder das Team aus verschiedenen Gründen, ersetzt durch neue Entwickler, die jeweils unterschiedliche Coding-Stile und bevorzugte Frameworks mitbrachten. Das neue Team entdeckt, dass kritische Betrugserkennungsregeln nie dokumentiert wurden — sie wurden basierend auf mündlichen Vereinbarungen und institutionellem Wissen implementiert, das mit den ursprünglichen Entwicklern ging. Als eine neue Regulierung Aktualisierungen der Betrugserkennungslogik erfordert, verbringt das aktuelle Team Wochen damit, die bestehenden Regeln zurückzuentwickeln, weil niemand versteht, warum bestimmte Entscheidungen getroffen wurden. Zusätzlich enthält die Codebasis jetzt drei verschiedene Ansätze zur Fehlerbehandlung, zwei verschiedene Logging-Frameworks und inkonsistente Datenbankzugriffsmuster, was Wartung zunehmend schwierig macht. Ein weiteres Beispiel betrifft eine Datenanalyse-Plattform, bei der der Abgang des ursprünglichen Architekten zu sechs Monaten verringerter Produktivität führte, während das verbleibende Team darum kämpfte, das komplexe Design der Datenverarbeitungspipeline ohne Dokumentation oder die Möglichkeit zu klärenden Fragen zu verstehen.
