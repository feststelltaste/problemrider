---
title: Kontinuierliche Abhängigkeitsaktualisierungen
description: Abhängigkeits-Upgrades in kleinen automatisierten Schritten übernehmen,
  sobald sie veröffentlicht werden, sodass sie sich nie zu einer Migration anhäufen,
  die niemand anzugehen wagt.
category:
- Dependencies
- Process
- Security
problems:
- dependency-version-conflicts
- obsolete-technologies
- vendor-dependency-entrapment
- technology-lock-in
- shared-dependencies
- high-technical-debt
- increasing-brittleness
- legacy-skill-shortage
- regulatory-compliance-drift
- fear-of-breaking-changes
- maintenance-cost-increase
- api-versioning-conflicts
- technology-stack-fragmentation
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: continuous-dependency-updates
related_solutions:
- slug: dependency-management-strategy
  similarity: 0.85
- slug: ci-cd-pipeline
  similarity: 0.7
- slug: automated-code-migration
  similarity: 0.7
- slug: development-workflow-automation
  similarity: 0.7
- slug: continuous-deployment
  similarity: 0.7
- slug: regular-maintenance-and-updates
  similarity: 0.7
---

## Description

Kontinuierliche Abhängigkeitsaktualisierungen bedeuten, Abhängigkeiten in kleinen Schritten zu aktualisieren, automatisch vorgeschlagen sobald neue Versionen veröffentlicht werden, statt in periodischen Großaufwänden. Der Mechanismus ist ein Bot wie Renovate oder Dependabot, der pro Update einen Pull Request öffnet, eine Pipeline, die ihn verifiziert, und eine Teamgewohnheit, die routinemäßigen zügig zu mergen. Beachten Sie, was ein solcher Bot nicht tut: Er hebt die Version an und lässt Ihren Quellcode unberührt, sodass dort, wo die neue Version eine API geändert hat, der Build bricht und die Anpassung der Aufrufstellen ein separater Aufwand ist — dafür ist automatisierte Code-Migration da. Sein Wert liegt vollständig in der Kumulation. Eine Abhängigkeit, die innerhalb von Wochen nach jedem Release aktualisiert wird, ist eine Serie kleiner, einzeln trivialer Änderungen; dieselbe Abhängigkeit, vier Jahre lang liegengelassen, ist eine einzelne Migration über mehrere Major-Versionen hinweg, mit angehäuften Breaking Changes, unklarer Fehlerfläche und einem Aufwandsschätzwert, der groß genug ist, um erneut aufgeschoben zu werden. Jede Legacy-Codebasis, die gefährlich hinter ihren Abhängigkeiten zurückliegt, ist auf dieselbe Weise dorthin gekommen — nicht durch eine Entscheidung, sondern durch das Fehlen einer solchen, wöchentlich wiederholt über Jahre.

## How to Apply ◆

> Der Grund, warum eine Codebasis fünf Major-Versionen zurückliegt, ist nie, dass jemand entschieden hat, es zu sein; es ist, dass kein Upgrade jemals dringend genug war, um es einzuplanen.

- **Automatisieren Sie den Vorschlag, nicht den Merge.** Ein Werkzeug wie Renovate oder Dependabot, das pro Update einen Pull Request öffnet, entfernt den Schritt, der Upgrades tatsächlich blockiert — nämlich dass niemand bemerkt, dass eine neue Version existiert.
- **Trennen Sie das Routinemäßige vom Bedeutsamen.** Patch- und Minor-Updates gut gepflegter Abhängigkeiten können bei grüner Pipeline mit wenig Zeremonie gemergt werden; Major-Versionen und alles, was ein Framework betrifft, brauchen eine menschliche Entscheidung. Alle Updates identisch zu behandeln erzeugt entweder gefährliche Automatisierung oder einen Rückstau ignorierter Pull Requests.
- **Bündeln Sie die lauten.** Alle Patch-Updates in einem wöchentlichen Pull Request zu gruppieren hält das Volumen handhabbar. Zwanzig einzelne Pull Requests pro Woche werden ignoriert, und ignorierte Update-Pull-Requests sind schlimmer als keine, weil sie das Ignorieren normalisieren.
- **Machen Sie die Pipeline zum Torwächter.** Diese Praxis hängt vollständig von der Testsuite ab; wo die Abdeckung dünn ist, verbreitet die Automatisierung Fehler, statt sie zu verhindern. In einer Legacy-Codebasis ist die Etablierung eines grundlegenden Sicherheitsnetzes um die kritischen Pfade die Voraussetzung, nicht eine optionale Verfeinerung.
- **Setzen Sie ein Veraltungsbudget** und behandeln Sie Überschreitungen als Arbeit: keine Produktionsabhängigkeit mehr als zwei Minor-Versionen oder sechs Monate zurück, mit dokumentierten Ausnahmen. Ohne ein festgelegtes Limit häufen sich die Pull Requests an, und die Praxis wird stillschweigend zur Dekoration.
- **Behandeln Sie die Majors bewusst**, eine nach der anderen, mit gelesenen Release Notes und einem angewandten Rezept, wo eines existiert. Dies sind die Updates, die Breaking Changes tragen, und hier treffen sich automatisierte Code-Migration und diese Praxis.
- **Verfolgen Sie End-of-Support-Daten zentral** neben Versionen. Aktuell zu sein ist nicht dasselbe wie unterstützt zu sein, und eine Abhängigkeit, deren Maintainer aufgehört hat, ist ein anderes Problem, das kein Update-Tool aufdecken wird.
- **Führen Sie es in einer ruhigen Phase ein**, nicht unter Lieferdruck. Die ersten Wochen erzeugen einen Rückstau angehäufter Updates, der abgearbeitet werden muss, und dies unter laufender Auslieferung zu tun erzeugt einen schlechten ersten Eindruck der Praxis.
- **Beobachten Sie die Lieferkette.** Häufige automatische Updates erhöhen die Exposition gegenüber kompromittierten Paketen, also fixieren Sie Versionen, verifizieren Sie Integrität und bevorzugen Sie eine kurze Verzögerung nach Veröffentlichung gegenüber dem Mergen innerhalb von Minuten nach Release.

## Tradeoffs ⇄

> Klein und häufig hält Upgrades trivial und schließt Sicherheitslücken schnell, aber es hängt von einer Testsuite ab, die Legacy-Codebasen oft fehlt, und es verbraucht kontinuierlich Aufmerksamkeit.

**Vorteile:**

- Upgrades bleiben klein und einzeln trivial, was die Anhäufung verhindert, die sie zu Migrationen macht, die niemand genehmigen will.
- Sicherheitspatches kommen innerhalb von Tagen statt beim nächsten Audit an, was in den meisten Organisationen der mit Abstand größte praktische Nutzen ist.
- Die Codebasis bleibt innerhalb des unterstützten Fensters, sodass Hilfe, Dokumentation und Einstellung von Personal für die verwendeten Versionen alle verfügbar bleiben.
- Breaking Changes werden einzeln angetroffen, mit den relevanten Release Notes, statt mehrerer Versionen auf einmal ohne klare Zuordnung.
- Der Upgrade-Aufwand wird zu einem vorhersehbaren kleinen Overhead statt einem gelegentlichen großen Projekt, das gerechtfertigt werden muss.

**Kosten und Risiken:**

- Es hängt von einer Testsuite ab, die gut genug ist, um zu erkennen, was ein Upgrade bricht; ohne eine solche ist die Automatisierung ein Mechanismus zum Ausliefern von Regressionen.
- Der Strom von Pull Requests verbraucht jede Woche Aufmerksamkeit, und Teams, die mit der Überprüfung zurückfallen, enden mit einem Rückstau, der die Praxis diskreditiert.
- Häufige automatische Updates erweitern die Angriffsfläche der Lieferkette, und ein kompromittiertes Paket kann Produktion schneller erreichen, als es sonst würde.
- Manche Legacy-Stacks haben Abhängigkeiten, die wirklich nicht aktualisiert werden können — eine fixierte Laufzeitumgebung, eine herstellerzertifizierte Version —, und das Tooling wird weiterhin Änderungen vorschlagen, die dauerhaft abgelehnt werden müssen.
- Die Einführung auf einer stark veralteten Codebasis erzeugt eine anfängliche Flut, die echte Arbeit ist und der Punkt, an dem die meisten Versuche aufgegeben werden.

## How It Could Be

Der Java-Service eines Teams hatte 84 Abhängigkeiten, von denen 31 mehr als zwei Jahre zurücklagen und vier veröffentlichte kritische Schwachstellen hatten, die monatelang unbemerkt geblieben waren. Ihr Upgrade-Ansatz war ein jährlicher Aufwand gewesen, der in zwei der letzten drei Jahre abgesagt wurde. Sie führten automatisierte Update-Pull-Requests ein, gruppierten Patch-Updates wöchentlich und setzten eine Regel, dass eine grüne Pipeline plus ein Reviewer für Patch- und Minor-Versionen ausreichend war. Die ersten sechs Wochen waren unangenehm: etwa 40 angehäufte Updates, die abgearbeitet werden mussten, drei davon brachen den Build auf Weisen, die jeweils einen Tag zur Diagnose brauchten. Danach war der stabile Zustand zwei bis drei Pull Requests pro Woche, meist innerhalb eines Tages gemergt. Achtzehn Monate später lag die älteste Abhängigkeit vier Monate zurück, und die Zeit von einer Sicherheitswarnung bis zu einem gepatchten Produktions-Deployment war von einer unbemessenen Anzahl von Monaten auf einen Median von drei Tagen gesunken.

Die Voraussetzung erwies sich als die schwierigere Hälfte. Ihr erster Versuch dabei war nach zwei Wochen aufgegeben worden, weil das Mergen von Updates Produktion zweimal brach — die Testsuite deckte etwa 20 Prozent des Codes und keine der Integrationspfade ab. Der zweite Versuch begann damit, Charakterisierungstests um die vier kritischen Abläufe zu schreiben, was drei Wochen dauerte und die eigentlichen Kosten der Praxis waren. Die Einschätzung des Teams danach war, dass sie drei Wochen für Tests aufgewendet hatten, um eine Abhängigkeitspraxis lebensfähig zu machen, und dabei auch das Sicherheitsnetz erworben hatten, das jede andere Art von Änderung in diesem Service weniger beängstigend machte.
