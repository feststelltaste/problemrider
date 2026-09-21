---
title: Feature Toggles
description: Aktivieren und Deaktivieren von Features für flexible Rollouts.
category:
- Process
- Operations
problems:
- large-risky-releases
- deployment-risk
- fear-of-change
- feature-creep
- release-instability
- frequent-hotfixes-and-rollbacks
- long-release-cycles
- development-disruption
- increased-time-to-market
- large-pull-requests
- strangler-fig-pattern-failures
- excessive-customization
layout: solution
lang: de
en_slug: feature-toggles
related_solutions:
- slug: feature-flags
  similarity: 0.9
- slug: dark-launches
  similarity: 0.75
- slug: rollback-mechanisms
  similarity: 0.75
- slug: canary-releases
  similarity: 0.7
- slug: restore-points
  similarity: 0.7
- slug: trunk-based-development
  similarity: 0.7
---

## Description

Feature Toggles sind bedingte Schalter — gehalten in einer Konfigurationsdatei, einem Datenbank-Flag oder einem dedizierten Feature-Flag-Dienst —, die neue Funktionalität umhüllen, sodass sie in deaktiviertem Zustand in die Produktion deployt und später, unabhängig vom Deployment selbst, aktiviert werden kann, was den Akt des Ausliefernns von Code vom Akt seines Releases an Nutzer entkoppelt. Dies ist besonders wertvoll für Legacy-System-Änderungen, die keine Big-Bang-Umschaltung vertragen: Ein Toggle lässt sowohl einen alten als auch einen neuen Codepfad gleichzeitig in der Produktion koexistieren, sodass eine riskante Neufassung von Legacy-Logik zu einem kleinen Prozentsatz des Verkehrs geleitet, mit dem bestehenden Verhalten verglichen und sofort durch Umlegen des Toggles zurückgenommen werden kann statt durch ein Notfall-Redeployment. Weil der Toggle als sofortiger Notschalter fungiert, gewinnen Teams das Vertrauen, Legacy-Änderungen vorzunehmen, die sich sonst zu riskant anfühlen würden, um sie zu versuchen, und prozentsatzbasierte Rollouts oder Nutzersegment-Targeting erlauben, eine Änderung mit echtem Produktionsverkehr zu validieren, bevor sie den Legacy-Pfad vollständig ersetzt. Der Zielkonflikt ist, dass in der Codebasis über ihre Nutzungsdauer hinaus verbleibende Toggles zu einer eigenen Form technischer Schulden werden, die Zahl möglicher Toggle-Zustandskombinationen erschöpfendes Testen unpraktikabel macht und jeder Toggle einen bedingten Zweig hinzufügt, der bereits komplexen Legacy-Code schwerer lesbar macht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie einen einfachen Toggle-Mechanismus ein (Konfigurationsdatei, Datenbank-Flags oder Feature-Flag-Dienst), bevor Sie komplexe Lösungen übernehmen
- Umhüllen Sie neue Funktionalität in bedingten Blöcken, die vom Toggle gesteuert werden, statt separate Code-Zweige zu pflegen
- Nutzen Sie Toggles, um Deployment vom Release zu entkoppeln, damit Code in deaktiviertem Zustand in die Produktion gelangen kann
- Implementieren Sie Notschalter für riskante Legacy-System-Änderungen, die sofortiges Rollback ohne erneutes Deployment erlauben
- Etablieren Sie einen Lebenszyklus für jeden Toggle: Definieren Sie, wann er entfernt wird, und räumen Sie veraltete Toggles regelmäßig auf
- Nutzen Sie prozentsatzbasierte Rollouts oder Nutzersegment-Targeting, um Änderungen zunächst mit einer Teilmenge des Verkehrs zu testen
- Überwachen Sie Schlüsselmetriken pro Toggle-Zustand, um Regressionen vor vollständiger Aktivierung zu erkennen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht inkrementelles Rollout von Änderungen in Legacy-Systemen mit verringertem Explosionsradius
- Entkoppelt Deployment vom Release und senkt die Deployment-Angst
- Erlaubt schnelles Rollback problematischer Features ohne Codeänderungen
- Unterstützt A/B-Tests und Canary-Releases in Systemen ohne moderne Deployment-Infrastruktur

**Kosten und Risiken:**
- Toggle-Proliferation erzeugt kombinatorische Testherausforderungen und Codekomplexität
- Veraltete, in der Codebasis verbleibende Toggles werden selbst zu technischen Schulden
- Das Testen aller Toggle-Kombinationen ist unpraktikabel, was das Risiko ungetesteter Pfade erhöht
- Fügt bedingte Verzweigung hinzu, die Legacy-Code schwerer verständlich machen kann

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Versicherungsunternehmen musste seine Schadenbearbeitungslogik von einer Legacy-Regel-Engine zu einer neuen Implementierung migrieren, konnte sich aber weder eine längere Ausfallzeit noch eine Big-Bang-Umschaltung leisten. Das Team führte Feature Toggles ein, die es sowohl dem alten als auch dem neuen Verarbeitungspfad erlaubten, in der Produktion zu koexistieren. Sie leiteten zunächst 5 % der Schäden über den neuen Pfad, verglichen die Ausgaben und erhöhten den Prozentsatz über mehrere Wochen schrittweise. Als ein subtiler Berechnungsunterschied entdeckt wurde, deaktivierten sie den neuen Pfad innerhalb von Minuten, behoben das Problem und setzten den Rollout ohne jegliche kundenseitig sichtbare Auswirkung fort.
