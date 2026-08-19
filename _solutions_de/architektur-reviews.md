---
title: Architektur-Reviews
description: Systematische und regelmäßige Überprüfung der Softwarearchitektur.
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-reviews/
problems:
- insufficient-design-skills
- misunderstanding-of-oop
- procedural-background
- suboptimal-solutions
- complex-implementation-paths
- uncontrolled-codebase-growth
- single-entry-point-design
- high-coupling-low-cohesion
- gold-plating
- second-system-effect
- cargo-culting
- rapid-prototyping-becoming-production
- convenience-driven-development
- accumulated-decision-debt
layout: solution
lang: de
en_slug: architecture-reviews
related_solutions:
- slug: code-review-process-reform
  similarity: 0.85
- slug: architecture-decision-records
  similarity: 0.8
- slug: lightweight-design-review
  similarity: 0.8
- slug: architecture-review-board
  similarity: 0.8
- slug: architecture-conformity-analysis
  similarity: 0.8
- slug: architecture-workshops
  similarity: 0.75
---

## Description

Ein Architektur-Review ist eine strukturierte, wiederkehrende Bewertung des Designs eines Systems — unter Nutzung eines Formats wie ATAM oder einer einfachen Checkliste statt freier Debatte —, die vorgeschlagene und bestehende Entscheidungen gegen tatsächliche Qualitätsanforderungen und Geschäftsbedürfnisse prüft. In Legacy-Systemen dient dies einem doppelten Zweck: Es fängt Kopplung, Überengineering und cargo-kultierte Muster ab, während die Kosten ihrer Behebung noch niedrig sind, und es baut Designurteilsvermögen im Team auf, indem die Begründung hinter jeder Entscheidung explizit und für Herausforderung offen gemacht wird. Die Einbeziehung teamexterner Reviewer und die Anwendung desselben Maßstabs sowohl auf neue Vorschläge als auch auf die Trajektorie des bestehenden Systems verhindert, dass das Review lediglich die Annahmen verstärkt, die das aktuelle Design überhaupt erst hervorgebracht haben.

## How to Apply ◆

> In Legacy-Systemen dienen Architektur-Reviews einem doppelten Zweck: Sie fangen Designprobleme ab, bevor sie sich verfestigen, und sie bauen die Designfähigkeiten auf, die das Team braucht, um aufzuhören, diese Probleme überhaupt erst zu erzeugen. Der Review-Prozess selbst ist ein Lehrmechanismus.

- Planen Sie Architektur-Reviews in zwei Rhythmen: leichtgewichtige Reviews vor der Implementierung jedes Features, das mehr als drei Module berührt oder eine neue Komponente einführt, und umfassende Reviews vierteljährlich zur Bewertung der Gesamttrajektorie der strukturellen Gesundheit des Systems.
- Nutzen Sie ein strukturiertes Review-Format statt freier Diskussion, um zu verhindern, dass Reviews zu subjektiven Debatten werden. Die ATAM (Architecture Tradeoff Analysis Method) bietet ein bewährtes Format zur Bewertung architektonischer Entscheidungen gegen Qualitätsattribut-Szenarien, aber selbst eine einfache Checkliste, die Kopplung, Kohäsion, Trennung von Belangen und angemessene Musternutzung abdeckt, ist effektiv.
- Beziehen Sie mindestens einen teamexternen Reviewer für umfassende Reviews ein. Interne Reviewer teilen dieselben Annahmen und blinden Flecken wie die Entwickler; eine externe Perspektive identifiziert Muster wie Cargo Culting, Gold Plating oder den Second-System-Effekt, die das Team möglicherweise nicht erkennt, weil es den Entscheidungen zu nahe steht.
- Überprüfen Sie architektonische Entscheidungen gegen die tatsächlichen Geschäftsanforderungen, denen sie dienen, nicht gegen abstrakte „Best Practices". Dies verhindert direkt Gold Plating und den Second-System-Effekt, indem das Team gezwungen wird, jedes architektonische Element in Bezug auf ein konkretes Geschäftsbedürfnis zu rechtfertigen.
- Machen Sie Architektur-Reviews zu einer Lerngelegenheit, indem Sie Entwickler bitten, ihre Design-Begründung zu erklären. Wenn Entwickler mit prozeduralem Hintergrund oder unzureichenden Designfähigkeiten ihre Designs vorstellen, lehrt die Review-Diskussion selbst sie über alternative Ansätze, Designprinzipien und die beteiligten Abwägungen.
- Bewerten Sie, ob die Komplexität der vorgeschlagenen Lösung der Komplexität des Problems entspricht, das sie löst. Wenn eine einfache CRUD-Operation durch drei Abstraktionsschichten, einen Event-Bus und ein benutzerdefiniertes Framework implementiert wird, sollte das Review die Proportionalität hinterfragen. Dies fängt Cargo Culting und CV-getriebenes Überengineering ab.
- Überprüfen Sie Prototyp-zu-Produktion-Übergänge explizit. Wenn ein Prototyp oder Proof-of-Concept für den Produktionseinsatz vorgeschlagen wird, sollte das Review ihn gegen Produktionsanforderungen bewerten, einschließlich Fehlerbehandlung, Sicherheit, Skalierbarkeit, Observability und operativer Wartbarkeit.
- Dokumentieren Sie Review-Ergebnisse als Architecture Decision Records (ADRs), die erfassen, was überprüft wurde, welche Alternativen erwogen wurden, was entschieden wurde und warum. Dies schafft ein institutionelles Gedächtnis, das verhindert, dass dieselben schlechten Entscheidungen wiederholt werden, und gibt neuen Teammitgliedern Einblick in die Design-Begründung des Systems.
- Verfolgen Sie architektonische Metriken über die Zeit — Kopplung zwischen Modulen, Komponentengröße, Abhängigkeitstiefe, Anzahl zirkulärer Abhängigkeiten — und überprüfen Sie Trends statt nur Momentaufnahmen. Ein einzelnes Review zeigt den aktuellen Zustand; Trendanalyse offenbart, ob sich das System verbessert oder verschlechtert.

## Tradeoffs ⇄

> Architektur-Reviews bieten den Aufsichtsmechanismus, der verhindert, dass Legacy-Systeme die Designprobleme anhäufen, die sie überhaupt erst zu Legacy-Systemen machen, aber sie erfordern organisatorisches Engagement und müssen gegen die Liefergeschwindigkeit abgewogen werden.

**Vorteile:**

- Fängt Designprobleme früh ab, wenn sie kostengünstig zu beheben sind, und verhindert die Anhäufung struktureller Probleme, die Legacy-Systeme teuer in der Wartung machen — ein während des Reviews entdeckter Designfehler kostet Stunden zur Behebung, während derselbe in Produktion entdeckte Fehler Monate kostet.
- Baut Designfähigkeiten im gesamten Team auf, indem Entwickler bei jedem Review architektonischem Denken, alternativen Ansätzen und Abwägungsanalyse ausgesetzt werden, was direkt unzureichende Designfähigkeiten und Missverständnisse von OOP angeht, die Legacy-Probleme erzeugen.
- Verhindert Cargo Culting und CV-getriebene Entwicklung, indem Teams verlangt wird, zu artikulieren, warum eine bestimmte Technologie oder ein bestimmtes Muster für ihren Kontext angemessen ist, wodurch Entscheidungen herausgefiltert werden, die über „es ist beliebt" oder „ich möchte es in meinem Lebenslauf haben" hinaus nicht gerechtfertigt werden können.
- Identifiziert Single-Entry-Point-Designs, God Objects und High-Coupling-Muster, bevor sie sich verfestigen, während die Kosten der Umstrukturierung noch handhabbar sind.
- Schafft Verantwortlichkeit für architektonische Entscheidungen durch dokumentierte Protokolle und macht sichtbar, wenn bequemlichkeitsgetriebene Abkürzungen gegenüber nachhaltigen Lösungen gewählt wurden.

**Kosten und Risiken:**

- Architektur-Reviews fügen dem Entwicklungsprozess Zeit hinzu, und wenn sie zu schwergewichtig oder zu häufig sind, verlangsamen sie die Lieferung ohne proportionalen Nutzen — der Prozess muss auf den Rhythmus des Teams und das Risikoprofil des Systems abgestimmt werden.
- Reviews, die von unerfahrenen Reviewern durchgeführt werden, könnten subtile Designprobleme nicht abfangen oder falsche Bedenken aufwerfen, was Reibung ohne Qualitätsverbesserung schafft. Effektive Reviews erfordern Reviewer, die genug Systeme gesehen haben, um Muster und Anti-Muster zu erkennen.
- Wenn Review-Feedback als Kritik statt als Lehre geliefert wird, könnten Entwickler defensiv werden und beginnen, so zu designen, dass sie Reviews bestehen, statt Probleme zu lösen — dieselbe Dynamik wie defensive Coding-Praktiken, verschoben auf die architektonische Ebene.
- Reviews können zu Gatekeeping-Mechanismen werden, die architektonische Entscheidungsfindung in einer kleinen Gruppe konzentrieren, was Engpässe schafft und das breitere Team entmachtet, statt seine Fähigkeiten aufzubauen.
- In sich schnell bewegenden Umgebungen könnte die durch Architektur-Reviews eingeführte Verzögerung mit Marktdruck kollidieren, und Teams könnten Reviews für „dringende" Arbeit umgehen, was oft genau die Arbeit ist, die am meisten Review benötigt.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Architektur-Reviews genutzt wurden, um Designprobleme in Legacy-Systemumgebungen zu verhindern und zu korrigieren.

Das Entwicklungsteam eines Einzelhandelsunternehmens schlug vor, sein monolithisches Auftragsmanagementsystem durch eine Microservices-Architektur aus 15 Services zu ersetzen, inspiriert von Konferenzvorträgen großer Technologieunternehmen. Während eines Architektur-Reviews bat ein externer Reviewer das Team, jeden vorgeschlagenen Service auf eine spezifische Geschäftsfähigkeit abzubilden und zu erklären, warum er unabhängig deploybar sein müsse. Die Übung offenbarte, dass nur vier der fünfzehn Services echte Unabhängigkeitsanforderungen hatten; die verbleibenden elf waren feingranulare Zerlegungen, die verteilte Komplexität ohne Geschäftsnutzen erzeugen würden. Das Review führte das Team zu einem modularen Monolithen mit vier klar abgegrenzten Modulen, der die benötigten Isolationsvorteile lieferte, ohne den operativen Overhead eines verteilten Systems. Das Team räumte später ein, dass es ohne das Review achtzehn Monate damit verbracht hätte, Infrastruktur für Services zu bauen, die nie unabhängig sein mussten.

Ein Regierungssoftwareprojekt führte vierteljährliche Architektur-Reviews durch, die Kopplungsmetriken über Releases hinweg verfolgten. Über drei Reviews zeigten die Metriken, dass die Kopplung zwischen dem Bürgerregistrierungsmodul und dem Anspruchsberechtigungsmodul stetig zunahm — von 12 modulübergreifenden Abhängigkeiten auf 34 über neun Monate. Das Review identifizierte, dass Entwickler bequeme Abkürzungen nahmen, indem sie direkt die Datenbanktabellen des jeweils anderen abfragten, statt die definierte API zu nutzen. Das Review resultierte in einer durch CI-Tooling durchgesetzten Architekturregel: Kein Modul durfte direkt auf das Datenbankschema eines anderen Moduls zugreifen. Die Kopplungsmetrik sank innerhalb von zwei Releases auf 8, und das Team des Anspruchsberechtigungsmoduls konnte anschließend seine Datenbankimplementierung ersetzen, ohne die Registrierung zu beeinträchtigen.

Ein Gesundheitstechnologieunternehmen nutzte Architektur-Reviews gezielt, um Designfähigkeitslücken in einem Team anzugehen, in dem die meisten Entwickler aus prozeduralen Programmierhintergründen kamen. Jedes Review beinhaltete ein „Design-Alternativen"-Segment, in dem der Reviewer präsentierte, wie dasselbe Problem mit unterschiedlichen Designansätzen gelöst werden könnte — die prozedurale Lösung des Teams mit einer objektorientierten Alternative vergleichend, Abwägungen erklärend statt den OOP-Ansatz vorzuschreiben. Über zwölf Monate und vierundzwanzig Reviews verschoben sich die Designmuster des Teams messbar: statische Utility-Klassen sanken von 60 % der neuen Klassen auf 15 %, und die Nutzung von Interfaces und Polymorphismus stieg entsprechend. Die Reviews fungierten als kontinuierliches architektonisches Mentoring, eingebettet in den Lieferprozess.
