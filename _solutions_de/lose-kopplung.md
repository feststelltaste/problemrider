---
title: Lose Kopplung
description: Minimierung von Abhängigkeiten zwischen Modulen, damit sich
  Änderungen nicht kaskadieren.
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/maintainability/loose-coupling/
problems:
- high-coupling-low-cohesion
- circular-references
- ripple-effect-of-changes
- complex-implementation-paths
- difficult-code-comprehension
- difficult-to-understand-code
- increased-cognitive-load
- cognitive-overload
- uncontrolled-codebase-growth
- inconsistent-behavior
- inconsistent-execution
- single-entry-point-design
layout: solution
lang: de
en_slug: loose-coupling
related_solutions:
- slug: abstraction
  similarity: 0.8
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.7
- slug: solid-principles
  similarity: 0.7
- slug: separation-of-concerns
  similarity: 0.7
---

## Description

Lose Kopplung minimiert die direkten Abhängigkeiten zwischen Modulen, sodass eine Änderung an einer Stelle nicht unvorhersehbar in andere kaskadiert, indem harte Referenzen und gemeinsam genutzter interner Zustand durch gut definierte Schnittstellen ersetzt werden. Legacy-Systeme sammeln über Jahre bequemer Abkürzungen das Gegenteil davon an — Komponenten greifen direkt in die Klassen anderer und teilen sich Datenbanktabellen —, bis eine einzelne Änderung wie das Hinzufügen eines neuen Versanddienstleisters gleichzeitig Auftragsverarbeitung, Rechnungsstellung und drei Datenbanktabellen anfassen muss. Schnittstellen zuerst an den Hotspots höchster Kopplung einzuführen, gefunden durch Abhängigkeitsgraph-Analyse statt Vermutung, verkleinert diesen Explosionsradius direkt, obwohl Entkopplung zu weit getrieben werden kann: Über einen gewissen Punkt hinaus fragmentiert sie nur Funktionalität, die echt zusammengehört, und tauscht Kopplung gegen eine andere Art Komplexität ein.

## How to Apply ◆

> Lose Kopplung in Legacy-Systemen zu erreichen erfordert systematische Identifikation und Reduktion unnötiger Abhängigkeiten zwischen Komponenten, indem direkte Referenzen durch gut definierte Schnittstellen und Kommunikationsmuster ersetzt werden.

- Identifizieren Sie die Hotspots höchster Kopplung, indem Sie Abhängigkeitsgraphen mit statischen Analysewerkzeugen analysieren. Fokussieren Sie anfängliche Entkopplungsbemühungen auf Komponenten mit den meisten eingehenden und ausgehenden Abhängigkeiten, da diese die größten Kaskadeneffekte verursachen, wenn sie geändert werden. In Legacy-Systemen sind dies oft zentrale „God Objects" oder Utility-Module, von denen alles abhängt.
- Führen Sie gut definierte Schnittstellen zwischen Komponenten ein, die derzeit über gemeinsamen internen Zustand oder direkte Klassenreferenzen kommunizieren. Definieren Sie Verträge, die spezifizieren, was jede Komponente bereitstellt und benötigt, ohne Implementierungsdetails offenzulegen. Dies erlaubt Komponenten, sich unabhängig weiterzuentwickeln, solange sie ihre Verträge einhalten.
- Wenden Sie Dependency Inversion an, sodass High-Level-Module nicht von Low-Level-Modulen abhängen, sondern beide von Abstraktionen. In der Praxis bedeutet dies, Abhängigkeiten über Konstruktoren oder Konfiguration zu injizieren statt sie direkt zu instanziieren, was es ermöglicht, Implementierungen zu ersetzen, ohne Konsumenten zu ändern.
- Brechen Sie zyklische Abhängigkeiten, indem Sie die Abhängigkeitszyklen identifizieren (mit Werkzeugen wie JDepend, deptrac oder sprachspezifischen Analysatoren) und gemeinsame Konzepte in separate Module extrahieren, von denen beide Seiten abhängen können, ohne sich gegenseitig zu referenzieren. Dies adressiert zirkuläre Referenzprobleme direkt auf architektonischer Ebene.
- Nutzen Sie event-getriebene Kommunikation oder Message Passing für Interaktionen, die keine sofortige Antwort erfordern. Statt dass Modul A Modul B direkt aufruft, veröffentlicht Modul A ein Event, das Modul B abonniert. Dies entkoppelt den Sender vom Empfänger und erlaubt, neue Konsumenten hinzuzufügen, ohne den Publisher zu ändern.
- Etablieren Sie klare Modulgrenzen, die sich an Geschäftsfähigkeiten oder Bounded Contexts ausrichten. Jedes Modul sollte seine Daten besitzen und sie nur über seine öffentliche Schnittstelle offenlegen. Gemeinsam genutzte Datenbanken oder direkter Tabellenzugriff über Modulgrenzen hinweg gehören zu den hartnäckigsten Kopplungsquellen in Legacy-Systemen.
- Implementieren Sie eine Anti-Corruption-Schicht bei der Integration mit Legacy-Komponenten, die nicht sofort refaktoriert werden können. Diese Übersetzungsschicht isoliert den Rest des Systems von der Schnittstelle und den Datenstrukturen der Legacy-Komponente und verhindert, dass sich Legacy-Kopplung auf neuen Code ausbreitet.
- Übernehmen Sie inkrementelle Strangler-Fig-Entkopplung für große Legacy-Monolithen. Statt eine vollständige Umstrukturierung zu versuchen, extrahieren Sie ein gut abgegrenztes Stück nach dem anderen hinter eine saubere Schnittstelle und leiten Verkehr durch die neue Schnittstelle, während die Legacy-Implementierung schrittweise ersetzt wird.

## Tradeoffs ⇄

> Lose Kopplung macht Systeme modularer und änderungsresistenter, führt aber Indirektion ein und erfordert diszipliniertes Schnittstellendesign, das Vorabaufwand summiert.

**Vorteile:**

- Verringert den Kaskadeneffekt von Änderungen dramatisch, indem Modifikationen auf die geänderte Komponente beschränkt werden, statt sich über das System zu kaskadieren.
- Verringert kognitive Last, weil Entwickler einzelne Komponenten verstehen und ändern können, ohne das gesamte System begreifen zu müssen.
- Macht Code leichter testbar, weil lose gekoppelte Komponenten isoliert mit Mock- oder Stub-Implementierungen ihrer Abhängigkeiten getestet werden können.
- Ermöglicht parallele Entwicklung, indem Teams erlaubt wird, gleichzeitig an verschiedenen Komponenten zu arbeiten, ohne Merge-Konflikte oder Integrationsprobleme zu erzeugen.
- Unterstützt inkrementelle Modernisierung, weil einzelne Komponenten unabhängig ersetzt oder aktualisiert werden können, ohne systemweite Änderungen zu erfordern.
- Verringert unkontrolliertes Codebasis-Wachstum, indem modulares Design gefördert wird, bei dem neue Features spezifische Komponenten erweitern, statt sich über die gesamte Codebasis zu verteilen.

**Kosten und Risiken:**

- Führt Indirektion ein, die es schwerer machen kann, den Ausführungsfluss durch das System zu verfolgen, besonders für Entwickler, die mit den genutzten Entkopplungsmustern nicht vertraut sind.
- Event-getriebene Architekturen fügen Komplexität beim Debugging und der Sicherstellung von Konsistenz hinzu, da der Kontrollfluss weniger explizit ist als bei direkten Methodenaufrufen.
- Die Definition stabiler Schnittstellen erfordert Vorabdesign-Aufwand, und schlecht gestaltete Schnittstellen können selbst zu einer Quelle von Starrheit werden, wenn sie häufige Änderungen benötigen.
- Übermäßige Entkopplung kann Funktionalität fragmentieren, die echt zusammengehört, was Kohäsion verringert, während Kopplungsreduktion verfolgt wird. Das Ziel ist angemessene Kopplung, nicht null Kopplung.
- In Legacy-Systemen bedeutet die inkrementelle Einführung loser Kopplung, dass die Codebasis vorübergehend sowohl eng gekoppelte als auch lose gekoppelte Abschnitte enthält, was Entwickler während der Übergangsphase verwirren kann.
- Performance-sensible Pfade tolerieren möglicherweise nicht den Overhead von Indirektionsschichten, Nachrichtenserialisierung oder Netzwerkaufrufen, die durch Entkopplungsmuster eingeführt werden.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Techniken loser Kopplung häufige Legacy-System-Probleme adressieren.

Ein Logistikunternehmen hat ein monolithisches Auftragsverwaltungssystem, in dem die Module Auftragsverarbeitung, Bestandsverfolgung, Versandberechnung und Rechnungserzeugung sich alle gegenseitig direkt auf ihre internen Klassen referenzieren und Datenbanktabellen teilen. Das Hinzufügen eines neuen Versanddienstleisters erfordert Änderungen im Auftragsverarbeitungsmodul, dem Versandmodul, den Rechnungsvorlagen und drei Datenbanktabellen. Das Team führt definierte Schnittstellen zwischen diesen Modulen ein, beginnend mit der Versandberechnungskomponente. Sie erstellen eine `ShippingProvider`-Schnittstelle, die das Auftragsverarbeitungsmodul aufruft, ohne zu wissen, welche Dienstleister-Implementierung die Anfrage bearbeitet. Neue Dienstleister werden hinzugefügt, indem diese Schnittstelle implementiert und die Implementierung registriert wird, ohne Auftragsverarbeitungs- oder Rechnungscode anzufassen. Über sechs Monate wendet das Team dasselbe Muster auf Bestand und Rechnungsstellung an und verringert die durchschnittliche Zahl geänderter Dateien pro Feature von 23 auf 6.

Eine Finanzdienstleistungsanwendung leidet unter zirkulären Abhängigkeiten zwischen ihren Modulen Kontoverwaltung, Transaktionsverarbeitung und Reporting. Das Kontomodul referenziert Transaktionsklassen, um aktuelle Aktivität anzuzeigen, während das Transaktionsmodul Kontoklassen referenziert, um Salden zu validieren, und das Reporting-Modul beide referenziert. Jedes einzelne Modul zu ändern erfordert einen Neubau aller drei. Das Team extrahiert gemeinsame Domänenkonzepte — Kontoidentifikatoren, Transaktionszusammenfassungen und Saldo-Snapshots — in ein schlankes gemeinsam genutztes Kernel-Modul, das nur Data Transfer Objects und Schnittstellen enthält. Jedes Modul hängt vom gemeinsamen Kernel ab, aber nicht voneinander. Die zirkuläre Abhängigkeit wird beseitigt, Build-Zeiten sinken von 12 Minuten auf 3 Minuten, und Entwickler können nun das Reporting-Modul ändern, ohne eine Neukompilierung von Kontoverwaltung oder Transaktionsverarbeitung auszulösen.
