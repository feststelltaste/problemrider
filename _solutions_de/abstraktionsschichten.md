---
title: Abstraktionsschichten
description: Kapselung hardwarespezifischer Details durch Abstraktionsschichten.
category:
- Architecture
- Code
problems:
- tight-coupling-issues
- technology-lock-in
- vendor-lock-in
- difficult-code-reuse
- hidden-dependencies
- architectural-mismatch
- abi-compatibility-issues
- dependency-on-supplier
layout: solution
lang: de
en_slug: abstraction-layers
related_solutions:
- slug: database-abstraction
  similarity: 0.85
- slug: protocol-abstraction
  similarity: 0.85
- slug: abstracted-file-system-access
  similarity: 0.85
- slug: abstraction
  similarity: 0.8
- slug: adapter
  similarity: 0.8
- slug: object-relational-mapping-orm
  similarity: 0.75
---

## Description

Abstraktionsschichten führen technologieneutrale Schnittstellen zwischen der Geschäftslogik und der Hardware, den Anbieter-SDKs oder plattformspezifischen APIs ein, von denen die Logik abhängt, sodass die konkrete Implementierung hinter der Schnittstelle ausgetauscht werden kann, ohne den Code zu berühren, der sie nutzt. Jede unterstützte Plattform oder jeder Anbieter erhält ihren eigenen Adapter, der die gemeinsame Schnittstelle implementiert, und Dependency Injection verdrahtet zur Laufzeit den korrekten Adapter basierend auf der Deployment-Umgebung. Legacy-Systeme häufen oft direkte Abhängigkeiten von der SDK eines einzelnen Anbieters oder einer bestimmten Hardwareplattform an, weil dies die einzige Option war, als das System gebaut wurde, und über Jahre verwandelt dies eine Geschäftsentscheidung eines einzelnen Anbieters — eine Preiserhöhung, eine End-of-Life-Ankündigung, eine Lizenzänderung — in ein existenzielles Risiko für das gesamte System. Durch das Zwischenschalten einer Abstraktionsschicht wird die Geschäftslogik unabhängig von jedem einzelnen Lieferanten, und eine Anbieter- oder Plattformmigration wird zu einer Frage des Schreibens eines neuen Adapters statt der Neuschreibung der Anwendung. Dies ist besonders wertvoll in der Legacy-Modernisierung, weil es erlaubt, die alte und neue Plattform während eines graduellen Umschaltens nebeneinander laufen zu lassen, statt einen riskanten Big-Bang-Ersatz zu erzwingen. Der Ansatz fügt tatsächlich eine Indirektionsschicht hinzu, sodass er typischerweise zuerst an den schmerzhaftesten und riskantesten Kopplungspunkten eingeführt wird, statt einheitlich über das gesamte System angewendet zu werden.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie plattform- oder anbieterspezifische Abhängigkeiten in der Legacy-Codebasis, die die Portabilität einschränken
- Definieren Sie technologieneutrale Schnittstellen, die die essenziellen Operationen erfassen, ohne Implementierungsdetails offenzulegen
- Implementieren Sie konkrete Adapter für jede Zielplattform oder Technologie hinter der Abstraktion
- Nutzen Sie Dependency Injection, um die passende Implementierung zur Laufzeit basierend auf der Deployment-Umgebung zu verdrahten
- Migrieren Sie Legacy-Code, um von den Abstraktionsschnittstellen statt konkreten Implementierungen abzuhängen
- Beginnen Sie mit den schmerzhaftesten Kopplungspunkten und erweitern Sie die Abstraktionsschicht inkrementell
- Testen Sie jeden Adapter unabhängig und verifizieren Sie, dass das Verhalten über Implementierungen hinweg konsistent ist

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Migration zwischen Plattformen, Anbietern oder Technologien, ohne Geschäftslogik neu zu schreiben
- Verbessert die Testbarkeit, indem Mock- oder In-Memory-Implementierungen erlaubt werden
- Verringert den Wirkungsradius von Technologieänderungen auf die Adapterschicht
- Fördert sauberere Architektur durch Trennung der Zuständigkeiten

**Kosten und Risiken:**
- Abstraktionsschichten fügen Indirektion hinzu, die verschleiern kann, was zur Laufzeit tatsächlich passiert
- Das richtige Abstraktionsniveau zu designen ist schwierig; zu breit und es leckt, zu eng und es überschränkt
- Die Wartung mehrerer Adapterimplementierungen erhöht die gesamte Wartungsfläche
- Verfrühte Abstraktion kann unnötige Komplexität hinzufügen, wenn Portabilität tatsächlich nicht benötigt wird
- Performance-kritische Pfade können unter der zusätzlichen Indirektion leiden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Legacy-Steuerungssystem eines Fertigungsunternehmens war eng an die proprietäre SDK eines bestimmten SPS-Anbieters (speicherprogrammierbare Steuerung) gekoppelt. Als der Anbieter das End-of-Life für seine Produktlinie ankündigte, stand das Team vor einer kompletten Neuschreibung. Stattdessen führten sie eine Hardware-Abstraktionsschicht ein, die generische Schnittstellen für Sensorablesung, Aktuatorsteuerung und Alarmverwaltung definierte. Sie implementierten Adapter sowohl für die SDK des bestehenden Anbieters als auch für die API des neuen Anbieters. Dies erlaubte ihnen, Produktionslinien inkrementell zu migrieren, wobei beide Hardwareplattformen während des Übergangs gleichzeitig liefen, und die Geschäftslogik blieb während des gesamten Prozesses vollständig unverändert.
