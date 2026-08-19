---
title: Anpassungen unter Versionskontrolle
description: Export der Konfiguration und individuellen Logik eines paketierten Systems
  in Textartefakte, sodass sie wie jeder andere Code diffbar, überprüfbar, rückgängig
  zu machen und deploybar sind.
category:
- Operations
- Process
- Code
problems:
- customization-outside-version-control
- low-code-customization-sprawl
- configuration-drift
- manual-deployment-processes
- configuration-chaos
- authorization-role-explosion
- lack-of-ownership-and-accountability
- invisible-nature-of-technical-debt
- regression-bugs
- slow-incident-resolution
- upgrade-blocked-by-customization
- inadequate-configuration-management
- core-modification-of-standard-software
- implementation-partner-dependency
layout: solution
lang: de
en_slug: customization-under-version-control
related_solutions:
- slug: ci-cd-pipeline
  similarity: 0.7
- slug: infrastructure-as-code
  similarity: 0.7
- slug: explicit-extension-points
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
- slug: self-service-developer-platform
  similarity: 0.65
- slug: development-workflow-automation
  similarity: 0.65
---

## Description

Anpassungen unter Versionskontrolle bedeutet, die Konfiguration und individuelle Logik eines paketierten Systems aus seinem internen Speicher in Textartefakte zu extrahieren, die in einem Repository gehalten werden, und diese Artefakte als die maßgebliche Quelle zu behandeln, aus der Umgebungen gebaut werden. Der Punkt ist nicht Ordentlichkeit. Versionskontrolle ist das Substrat, auf dem Review, Reproduzierbarkeit, Nachverfolgbarkeit und Rückgängigmachen alle beruhen, und keines davon kann auf Zustand angewandt werden, der nur innerhalb eines laufenden Systems existiert. Teams, die ihren eigenen Code an hohen Standards messen, betreiben ihre paketierten Systeme häufig ohne jede dieser Praktiken, nicht aus Nachlässigkeit, sondern weil die Plattform die Option nie präsentiert hat. Den Export zu etablieren ist das, was jede andere Praxis verfügbar macht; bis er existiert, kann keine Menge an Prozessdisziplin das kompensieren.

## How to Apply ◆

> Das Hindernis ist fast nie, dass die Plattform nicht exportieren kann, sondern dass niemand den Export je zur Quelle statt zu einem Backup gemacht hat.

- **Finden Sie heraus, was die Plattform bereits exportieren kann.** Die meisten Unternehmenspakete bieten Transport-, Migrations- oder Serialisierungsfunktionen, die für das Verschieben von Änderungen zwischen Umgebungen gedacht sind. Diese erzeugen üblicherweise etwas, das commitet werden kann, auch wenn das Format unangenehm ist.
- **Etablieren Sie die Autoritätsrichtung explizit.** Das Repository ist die Quelle und Umgebungen werden daraus gebaut, oder das System ist die Quelle und das Repository ist eine Aufzeichnung. Ersteres ist das Ziel; Letzteres ist ein legitimer erster Schritt, und die beiden zu verwechseln erzeugt ein Repository, dem niemand vertraut.
- **Beginnen Sie mit dem, was sich am meisten ändert und am meisten wehtut**, typischerweise Workflow-Definitionen, Skripte und Formularlogik. Der Versuch, die gesamte Konfigurationsfläche auf einmal unter Kontrolle zu bringen, erzeugt ein stockendes Projekt.
- **Machen Sie die exportierte Form so lesbar, wie es die Plattform erlaubt.** Wo der Export ein Binärformat oder ein undurchsichtiger Blob ist, investieren Sie in eine Konvertierung, die etwas Diffbares erzeugt — der Diff ist, wo der meiste Wert liegt, und ein unvergleichbares Artefakt liefert wenig.
- **Führen Sie Review vor dem Deployment ein**, nicht bevor die Änderung in einer Entwicklungsumgebung vorgenommen wird. Review zu verlangen, bevor überhaupt jemand experimentieren darf, entfernt den Hauptvorteil der Plattform; es vor allem zu verlangen, was Produktion erreicht, stellt die fehlende Kontrolle wieder her.
- **Automatisieren Sie die Beförderung zwischen Umgebungen** aus dem Repository. Manuelle Beförderung hält das Repository beratend, und ein beratendes Repository driftet innerhalb von Wochen.
- **Erkennen Sie Drift kontinuierlich**, indem Sie die laufende Konfiguration nach Zeitplan mit dem Repository vergleichen. Direkte Produktionsänderungen werden passieren; die Frage ist, ob sie bemerkt werden.
- **Beschränken Sie, wer Produktion direkt ändern darf**, und protokollieren Sie es, wenn sie es tun. Notfallzugriff ist legitim und sollte eine Aufzeichnung hinterlassen, die dazu auffordert, die Änderung ins Repository zurückzubringen.
- **Beweisen Sie es mit einem Neuaufbau.** Eine funktionierende Umgebung allein aus dem Repository zu rekonstruieren ist der einzige echte Test, ob die Quelle maßgeblich ist, und er scheitert meist beim ersten Mal aufschlussreich.

## Tradeoffs ⇄

> Paketierte Anpassung unter Versionskontrolle zu bringen stellt die Praktiken wieder her, die die Plattform entfernt hat, auf Kosten des Aufbaus eines Export- und Deployment-Pfads, den der Hersteller nicht bereitgestellt hat.

**Vorteile:**

- Änderungen werden überprüfbar, bevor sie Produktion erreichen, was die mit Abstand größte Qualitätsverbesserung in diesen Umgebungen ist.
- Das Anpassungsinventar wird auflistbar und durchsuchbar, sodass es gezählt, bewertet und reduziert werden kann.
- Das Rückgängigmachen einer Änderung wird möglich, statt eine Rekonstruktion aus dem Gedächtnis zu sein.
- Umgebungen können aus einem bekannten Zustand neu gebaut werden, was sowohl Disaster Recovery als auch die Fähigkeit, ein Upgrade realistisch zu testen, verändert.
- Autorenschaft und Historie machen es möglich zu fragen, warum etwas so konfiguriert ist, wie es ist, und einen Adressaten für die Antwort zu haben.

**Kosten und Risiken:**

- Der Export- und Beförderungspfad ist echte Ingenieursarbeit, die der Hersteller nicht unterstützt und die mit einem Release brechen kann.
- Manch ein Plattformzustand kann überhaupt nicht sinnvoll exportiert werden, sodass die Abdeckung partiell sein wird und die Grenze dokumentiert werden muss, sonst wird dem Repository mehr vertraut, als es verdient.
- Die Einführung von Review fügt Änderungen Latenz hinzu, an die Administratoren gewöhnt sind, sofort vorzunehmen, und dies wird als Verlust empfunden.
- Ein Repository, das von der Realität abdriftet, ist schlimmer als keines, weil Entscheidungen dagegen getroffen werden; es maßgeblich zu halten erfordert, dass die Drift-Erkennung tatsächlich befolgt wird.
- Exportierte Formate sind häufig umfangreich und maschinenorientiert, sodass Diffs groß und schwer lesbar sein können, selbst sobald sie existieren.

## How It Could Be

Eine IT-Service-Management-Plattform hatte ihre gesamte Workflow- und Skriptlogik in der Plattformdatenbank, gepflegt von vier Administratoren ohne Review-Schritt. Das Team baute einen Export von Skripten und Workflow-Definitionen in ein Repository, anfangs als nächtlichen Snapshot ohne Autorität — rein als Aufzeichnung. Allein das änderte innerhalb eines Monats etwas: Der Snapshot-Diff wurde zur Antwort auf „was hat sich geändert", was zuvor erfordert hatte, Leute zu fragen, und es deckte sofort 310 Fragmente auf, die auf nicht mehr existierende Felder verwiesen. Sechs Monate später wurde die Richtung umgekehrt, mit Beförderung in Produktion, aus dem Repository gesteuert, und direkten Produktionsänderungen beschränkt auf einen Break-Glass-Pfad, der protokollierte und alarmierte.

Der Neuaufbau-Test war, wo sich der Wert als unbestreitbar erwies. Ihr Disaster-Recovery-Plan hatte angenommen, dass eine Ersatzinstanz aus Dokumentation konfiguriert werden könnte. Der Versuch aus dem Repository dauerte zwei Tage und scheiterte an drei Kategorien von Zustand, die der Export nicht abdeckte — Integrationsanmeldedaten, geplante Job-Definitionen und eine Reihe von Plattform-Ebenen-Einstellungen. Alle drei wurden dann explizit behandelt, zwei durch Erweiterung des Exports und eine durch Dokumentation als manueller Schritt mit einer Checkliste. Die tatsächliche Wiederherstellungsposition der Organisation verbesserte sich erheblich, und der Befund kam aus einer Übung, die die vorherige Anordnung selbst zu versuchen unmöglich gemacht hatte.
