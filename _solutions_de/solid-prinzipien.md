---
title: SOLID-Prinzipien
description: Anwendung grundlegender Designprinzipien für objektorientierte
  Programmierung.
category:
- Code
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/solid-principles/
problems:
- high-coupling-low-cohesion
- circular-references
- hidden-side-effects
- ripple-effect-of-changes
- misunderstanding-of-oop
- procedural-background
- procedural-programming-in-oop-languages
- insufficient-design-skills
- single-entry-point-design
- convenience-driven-development
- defensive-coding-practices
- uncontrolled-codebase-growth
- increased-technical-shortcuts
- excessive-class-size
- over-reliance-on-utility-classes
- poor-encapsulation
- bloated-class
- global-state-and-side-effects
- god-object-anti-pattern
- monolithic-functions-and-classes
layout: solution
lang: de
en_slug: solid-principles
related_solutions:
- slug: separation-of-concerns
  similarity: 0.75
- slug: clean-code
  similarity: 0.75
- slug: design-by-contract
  similarity: 0.75
- slug: incremental-refactoring
  similarity: 0.75
- slug: facades
  similarity: 0.7
- slug: pattern-language
  similarity: 0.7
---

## Description

SOLID sammelt fünf objektorientierte Designprinzipien — Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion — in ein gemeinsames, konkretes Vokabular dafür, wie eine gut strukturierte Klasse tatsächlich aussieht. In Legacy-Systemen ist dies selten eine akademische Übung: eine God-Class, die im Laufe von Jahren an Feature-Ergänzungen ein Dutzend nicht verwandter Verantwortlichkeiten absorbiert hat, oder eine bedingte Kette, die bei jeder Hinzufügung einer Variante einen neuen Zweig erhält, sind beide direkte Verletzungen, die jede nachfolgende Änderung teurer machen als die letzte, und die genaue Benennung, welches Prinzip ein Codestück verletzt, verwandelt einen vagen "das fühlt sich falsch an"-Code-Review-Kommentar in einen umsetzbaren, spezifischen Vorschlag. Inkrementell angewendet, beginnend dort, wo der Schmerz aus einer Verletzung derzeit am größten ist, gibt SOLID Entwicklern mit prozeduralem Hintergrund einen strukturierten Weg in objektorientiertes Denken, obwohl seine mechanische Anwendung — eine Schnittstelle für jede Klasse, unabhängig davon, ob jemals eine zweite Implementierung existieren wird — ihre eigene Art unnötiger Indirektion schafft.

## How to Apply ◆

> In Legacy-Systemen sind SOLID-Prinzipien keine akademische Übung — sie sind der direkteste Weg, den Schmerz jeder Änderung zu reduzieren. Ihre inkrementelle Einführung, beginnend dort, wo der Schmerz am größten ist, verwandelt eine Codebasis, die sich Änderungen widersetzt, in eine, die sie aufnimmt.

- Beginnen Sie mit dem Single Responsibility Principle (SRP) in den Bereichen mit dem höchsten Änderungsaufkommen: Identifizieren Sie die God-Classes und Single-Entry-Point-Controller, die über Jahre an Feature-Ergänzungen Verantwortlichkeiten angehäuft haben. Extrahieren Sie eine Verantwortlichkeit nach der anderen in eine neue Klasse und behalten Sie die ursprüngliche Klasse als dünnen Koordinator, bis sie auf eine handhabbare Größe schrumpft.
- Wenden Sie das Open/Closed Principle (OCP) an, wenn Sie sich dabei ertappen, jedes Mal dieselbe Switch-Anweisung oder If-Else-Kette zu bearbeiten, wenn eine neue Variante hinzugefügt wird. Ersetzen Sie bedingte Verzweigung durch Polymorphie, indem Sie eine Schnittstelle und eine Implementierung pro Variante einführen, sodass neue Varianten hinzugefügt werden können, ohne bestehenden Code zu modifizieren.
- Nutzen Sie das Liskov Substitution Principle (LSP), um bestehende Vererbungshierarchien in der Legacy-Codebasis zu prüfen. Suchen Sie nach Unterklassen, die Methoden überschreiben, indem sie Exceptions werfen oder Parameter still ignorieren — dies sind LSP-Verletzungen, die versteckte Nebenwirkungen und unvorhersehbares Verhalten erzeugen. Ersetzen Sie defekte Hierarchien durch Komposition oder ordentlich gestaltete Schnittstellen.
- Setzen Sie das Interface Segregation Principle (ISP) durch, indem Sie große Schnittstellen aufteilen, die Implementierer zwingen, Stub-Implementierungen für Methoden bereitzustellen, die sie nicht benötigen. In Legacy-Systemen sind diese aufgeblähten Schnittstellen oft der Grund, warum Komponenten eng an Verträge gekoppelt werden, die sie nur teilweise erfüllen.
- Führen Sie das Dependency Inversion Principle (DIP) ein, indem Sie direkte Instanziierung von Abhängigkeiten durch Constructor Injection ersetzen. In Legacy-Code bedeutet dies oft, konkrete Abhängigkeiten hinter Schnittstellen zu verpacken, sodass hochrangige Geschäftslogik nicht mehr von niedrigrangigen Infrastrukturdetails wie spezifischen Datenbanktreibern oder E-Mail-Bibliotheken abhängt.
- Lehren Sie SOLID-Prinzipien durch Code-Review, nicht durch Vortrag. Weisen Sie beim Review von Legacy-Code-Modifikationen auf spezifische Verletzungen hin und schlagen Sie konkrete Refaktorierungsschritte vor. Dies baut Verständnis im Kontext der tatsächlichen Codebasis des Teams auf statt im Abstrakten.
- Adressieren Sie Code im prozeduralen Stil in OOP-Sprachen, indem Sie demonstrieren, wie SOLID-Prinzipien natürlich zu Objekten führen, die sowohl Zustand als auch Verhalten kapseln, und das Muster von Utility-Klassen ersetzen, die auf passiven Datenstrukturen operieren.
- Nutzen Sie statische Analyseregeln, um häufige SOLID-Verletzungen automatisch zu erkennen — Klassen mit zu vielen Abhängigkeiten (SRP), Methoden, die nicht verwandten Zustand ändern (SRP), und konkrete Klassenreferenzen, wo Schnittstellen genutzt werden sollten (DIP).

## Tradeoffs ⇄

> SOLID-Prinzipien bieten ein gemeinsames Design-Vokabular, das Legacy-Code inkrementell wartbarer macht, erfordern aber Urteilsvermögen darüber, wann und wie strikt jedes Prinzip anzuwenden ist.

**Vorteile:**

- Reduziert Kopplung, indem sichergestellt wird, dass jede Klasse einen einzigen Grund zur Änderung hat, was direkt den Blast-Radius von Modifikationen verkleinert und den Kaskadeneffekt verhindert, der Legacy-Änderungen so teuer macht.
- Macht Code vorhersagbarer, indem versteckte Nebenwirkungen beseitigt werden: Wenn jede Klasse eine klare, fokussierte Verantwortlichkeit hat, können Entwickler verstehen, was eine Funktion tut, allein aus ihrem Namen und ihrer Signatur, ohne durch nicht verwandtes Verhalten zu verfolgen.
- Bietet Entwicklern mit prozeduralem Hintergrund einen strukturierten Weg zu objektorientiertem Denken, indem konkrete Regeln statt abstrakter OOP-Philosophie angeboten werden.
- Verbessert Testbarkeit, weil Klassen, die von Abstraktionen statt konkreten Implementierungen abhängen, isoliert mit einfachen Test-Doubles getestet werden können.
- Schafft eine gemeinsame Designsprache für Code-Review-Diskussionen und ersetzt subjektives "das fühlt sich falsch an"-Feedback durch spezifische, umsetzbare Prinzipreferenzen.

**Kosten und Risiken:**

- Übermäßige Anwendung von SOLID-Prinzipien erzeugt eine Explosion kleiner Klassen und Schnittstellen, die genauso schwer zu navigieren sein kann wie der ursprüngliche monolithische Code — das Ergebnis ist Indirektion ohne Klarheit.
- Die Einführung von Schnittstellen und Dependency Injection in eine Legacy-Codebasis ohne IoC-Container erfordert Infrastrukturarbeit, bevor sich die Designvorteile materialisieren, und diese Arbeit konkurriert mit der Feature-Lieferung.
- Entwickler, die mit SOLID nicht vertraut sind, könnten schlechtere Designs produzieren, indem sie die Prinzipien mechanisch anwenden, ohne ihre Absicht zu verstehen — zum Beispiel, indem sie Ein-Methoden-Schnittstellen für jede Klasse erstellen, unabhängig davon, ob jemals mehrere Implementierungen existieren werden.
- Die Refaktorierung von Legacy-Code hin zu SOLID-Prinzipien ohne ausreichende Testabdeckung birgt dasselbe Risiko wie jede Refaktorierung: Verhalten könnte sich auf Weisen ändern, die erst in Produktion erkannt werden.
- In manchen Legacy-Kontexten ist der prozedurale Stil tatsächlich angemessen — Batch-Verarbeitungsskripte, Datenmigrations-Utilities und einfache CRUD-Operationen profitieren möglicherweise nicht von voller SOLID-Behandlung, und das Erzwingen von OOP-Mustern fügt Komplexität ohne Wert hinzu.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie SOLID-Prinzipien konkrete Designprobleme adressieren, die in Legacy-Systemen auftreten.

Ein Logistikunternehmen pflegte ein Auftragsverarbeitungssystem, bei dem eine einzelne `OrderService`-Klasse über acht Jahre auf 3.200 Zeilen angewachsen war. Sie validierte Aufträge, berechnete Versandkosten, wendete Rabattregeln an, aktualisierte Bestand, sandte E-Mail-Benachrichtigungen und protokollierte Audit-Trails. Jeder neue Versanddienstleister oder jede neue Rabattaktion erforderte eine Modifikation dieser Klasse, und Änderungen an der E-Mail-Formatierung brachen gelegentlich Rabattberechnungen aufgrund gemeinsam genutzter Instanzvariablen. Das Team wendete SRP an, indem es jede Verantwortlichkeit in ihre eigene Klasse extrahierte — `ShippingCalculator`, `DiscountEngine`, `InventoryUpdater`, `NotificationSender` und `AuditLogger` —, jede mit einer klar definierten Schnittstelle. Der `OrderService` wurde auf einen 60-Zeilen-Orchestrator reduziert. Als das Unternehmen im folgenden Quartal einen neuen Versanddienstleister hinzufügte, erforderte die Änderung das Hinzufügen einer neuen Klasse, die die `ShippingCalculator`-Schnittstelle implementierte, und null Modifikationen an bestehenden Klassen, gemäß OCP.

Eine Banking-Anwendung hatte eine `ReportGenerator`-Klasse, die für verschiedene Berichtstypen unterklassifiziert wurde, aber mehrere Unterklassen warfen `UnsupportedOperationException` für Methoden, die sie erbten, aber nicht sinnvoll implementieren konnten. Als ein Batch-Job über alle Berichtsgeneratoren iterierte und `generateSummary()` aufrief, schlugen einige Unterklassen still fehl, was unvollständige Berichte produzierte, die erst durch die Monatsendabstimmung erfasst wurden. Das Team erkannte dies als LSP-Verletzungen und strukturierte die Hierarchie in separate Schnittstellen um — `DetailedReport` und `SummaryReport` —, sodass sich jede Implementierung nur zu Fähigkeiten verpflichtete, die sie tatsächlich liefern konnte. Der Batch-Job wurde aktualisiert, um Schnittstellentypen zu prüfen, und die stillen Fehlschläge wurden vollständig beseitigt.

Ein Gesundheitssoftware-Unternehmen stellte Entwickler mit starkem C- und COBOL-Hintergrund ein, um ein Java-basiertes Patientenverwaltungssystem zu pflegen. Die Codebasis wurde von statischen Utility-Klassen dominiert — `PatientUtils`, `BillingUtils`, `ScheduleUtils` —, jede mit Dutzenden statischer Methoden, die auf einfachen Datenobjekten operierten. Das Hinzufügen neuer Patiententypen erforderte die Modifikation jeder Utility-Klasse. Das Team führte SOLID-Prinzipien durch eine Reihe von Pair-Programming-Sitzungen ein, in denen sie demonstrierten, wie DIP und ISP das Utility-Klassen-Muster durch ordentliche Domänenobjekte ersetzen konnten, die Verhalten kapselten. Über sechs Monate konvertierte das Team die am häufigsten modifizierten Utilities in Service-Klassen mit injizierten Abhängigkeiten, was die durchschnittliche Anzahl der pro Feature geänderten Dateien von vierzehn auf drei reduzierte.
