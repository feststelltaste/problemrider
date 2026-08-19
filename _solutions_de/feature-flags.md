---
title: Feature Flags
description: Aktivieren oder Deaktivieren von Funktionen über Konfigurationsschalter.
category:
- Operations
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/feature-toggles/
problems:
- fear-of-breaking-changes
- deployment-risk
- large-risky-releases
- release-instability
- frequent-hotfixes-and-rollbacks
- missing-rollback-strategy
- release-anxiety
- long-lived-feature-branches
- merge-conflicts
- fear-of-change
- fear-of-failure
- past-negative-experiences
layout: solution
lang: de
en_slug: feature-flags
related_solutions:
- slug: feature-toggles
  similarity: 0.9
- slug: strangler-fig-pattern
  similarity: 0.8
- slug: ci-cd-pipeline
  similarity: 0.75
- slug: blue-green-canary-deployments
  similarity: 0.75
- slug: small-change-batches
  similarity: 0.75
- slug: code-quality-gates
  similarity: 0.7
---

## Description

Ein Feature Flag trennt das Deployment von Code vom Release an Nutzer, indem es einer neuen Implementierung erlaubt, inaktiv in der Produktion hinter einem Konfigurationsschalter zu verweilen, bis sie verifiziert wurde, um dann sofort aktiviert — und ebenso sofort zurückgenommen — zu werden, ohne ein erneutes Deployment. Diese Entkopplung ist besonders wertvoll in der Legacy-Modernisierung, wo eine neue Implementierung von Legacy-Verhalten als inaktiver Release-Toggle ausgeliefert, schrittweise nach Prozentsatz oder Konto ausgerollt und sofort zurückgenommen werden kann, wenn ein undokumentierter Grenzfall im Produktionsmaßstab auftaucht — nichts davon könnte ein traditioneller Legacy-Rollback-Prozess in dieser Geschwindigkeit bieten. Das Risiko ist, dass Toggles, die nach Erfüllung ihres Zwecks bestehen bleiben, zu einer Form toten Codes werden, der wie beabsichtigte Konfiguration aussieht, sodass jedem Toggle bei der Erstellung eine maximale Lebensdauer zu geben und einen abgelaufenen als Bug zu behandeln, den Mechanismus davor bewahrt, sich in dieselbe Art von Schulden zu verwandeln, die er eigentlich beseitigen sollte.

## How to Apply ◆

> In der Legacy-Modernisierung entkoppeln Feature Flags den Akt des Deployments neuen Codes vom Akt, ihn Nutzern zugänglich zu machen, wodurch es möglich wird, Änderungen an einer fragilen Produktionsumgebung in kleinen, reversiblen Schritten auszuliefern.

- Nutzen Sie Release-Toggles, um neue Implementierungen von Legacy-Verhalten inaktiv in die Produktion auszuliefern; der alte Codepfad bleibt live, bis das Team den neuen unter realen Bedingungen verifiziert hat.
- Umhüllen Sie jede ersetzte Legacy-Komponente mit einem Ops-Toggle, der Verkehr sofort ohne erneutes Deployment auf den alten Code zurücklenken kann — dies ist das Sicherheitsnetz, das Teams erlaubt, sich schneller zu bewegen in Systemen, in denen Rollbacks traditionell langsam und riskant sind.
- Führen Sie Toggles an der Nahtstelle zwischen alter und neuer Implementierung ein statt Bedingungen über die Geschäftslogik zu verstreuen; nutzen Sie das Strategy-Pattern, damit der Toggle beim Start auswählt, welche Implementierung injiziert wird.
- Wenden Sie Prozentsatz-Rollouts an, um eine neue Implementierung schrittweise wachsenden Nutzeranteilen auszusetzen und das Verhalten anhand von Live-Daten zu validieren, bevor vollständig umgeschaltet wird — besonders wichtig, wenn das Legacy-System undokumentierte Grenzfälle hat, die erst im Produktionsmaßstab auftauchen.
- Koordinieren Sie den Toggle-Zustand mit den Release-Fenstern des Legacy-Systems; in Organisationen, in denen das Legacy-System vierteljährlich deployt, erlauben Toggles, dass neuer Servicecode kontinuierlich deployt wird, während das geschäftlich sichtbare Feature-Release separat gesteuert wird.
- Legen Sie für jeden Release-Toggle bei der Erstellung eine maximale Lebensdauer fest und behandeln Sie abgelaufene Toggles als Bugs; in Legacy-Kontexten werden bestehen bleibende Toggles zu permanentem totem Code, der genau wie die technischen Schulden aussieht, denen das Team zu entkommen versuchte.
- Protokollieren Sie, welcher Toggle-Zustand bei jeder Anfrage aktiv war, damit Diskrepanzen zwischen altem und neuem Verhalten während der Vorfalluntersuchung nachvollzogen werden können.
- Bevor Sie einen Toggle entfernen, lassen Sie beide Codepfade im Shadow-Modus laufen — führen Sie den neuen Pfad aus, aber vergleichen Sie seine Ausgabe mit dem Ergebnis des alten Pfads, ohne es zu verwenden —, um Parität vor der endgültigen Umschaltung zu verifizieren.

## Tradeoffs ⇄

> Feature Flags geben Legacy-Modernisierungsteams einen mächtigen Mechanismus für inkrementelle, reversible Änderung, aber sie müssen so sorgfältig verwaltet werden wie der Code, den sie schützen, sonst werden sie zu einer Quelle derselben Komplexität, die sie eigentlich verringern sollten.

**Vorteile:**

- Beseitigt die Notwendigkeit riskanter Big-Bang-Umschaltungen, indem neue Implementierungen mit Legacy-Code in der Produktion koexistieren und schrittweise aktiviert werden können.
- Ermöglicht sofortiges Rollback eines bestimmten neuen Features ohne erneutes Deployment, ohne dabei unabhängige Bugfixes oder andere Verbesserungen zurückzunehmen, die im selben Deployment ausgeliefert wurden.
- Erlaubt kontinuierliches Deployment von Code in eine Legacy-Produktionsumgebung, selbst wenn die geschäftliche Freigabe für ein Release noch Tage oder Wochen entfernt ist.
- Verringert den Explosionsradius eines Defekts in neuem Code, indem die Exposition auf einen kontrollierten Verkehrsanteil oder eine bestimmte Nutzergruppe vor vollständigem Rollout begrenzt wird.
- Gibt Betriebsteams eine Laufzeit-Kontrollfläche, um ressourcenintensive neue Features bei Spitzenlast zu deaktivieren, ohne auf eine Codeänderung zu warten.

**Kosten und Risiken:**

- Jeder aktive Toggle fügt einen Codepfad hinzu, der in beiden Zuständen getestet werden muss; Legacy-Systeme mit bereits schlechter Testabdeckung können schnell Kombinationen ansammeln, die niemand verifiziert hat.
- Toggles, die nach Erfüllung ihres Zwecks bestehen bleiben, sind eine Form technischer Schulden, die in Legacy-Codebasen besonders heimtückisch ist — sie sehen wie beabsichtigte Konfiguration aus, verbergen aber toten Code.
- In Organisationen mit manuellen Deployment-Prozessen und langen Release-Zyklen kann die Verwaltung des Toggle-Zustands zu einem Koordinationsproblem werden, wenn mehrere Teams unabhängig überlappende Toggles kontrollieren.
- Das Geschäfts- und Betriebspersonal, das den Toggle-Zustand während Vorfällen verwalten muss, hat in Legacy-Umgebungen oft kein passendes Tooling dafür, was dazu führt, dass Toggle-Änderungen als Ad-hoc-Anfragen über Entwickler geleitet werden.
- Verschachtelte Toggles — bei denen ein per Toggle gesteuerter Codepfad einen weiteren, von einem zweiten Toggle abhängigen enthält — erzeugen Interaktionskomplexität, über die man kaum noch nachdenken kann, ein Risiko, das in Systemen wächst, in denen mehrere Modernisierungsstränge gleichzeitig laufen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Feature Flags sicherere, kontrolliertere Änderung in der Legacy-System-Modernisierung ermöglicht haben.

Eine Behörde des öffentlichen Sektors ersetzte eine zwanzig Jahre alte Leistungsberechnungs-Engine durch eine neu implementierte Version basierend auf aktueller Gesetzgebung. Weil die alte Engine nie formal getestet worden war, gab es keine Möglichkeit, Äquivalenz vor der Inbetriebnahme zu verifizieren. Das Team umhüllte die neue Engine mit einem Release-Toggle und ließ beide Engines während einer Shadow-Periode parallel laufen: Jede Berechnungsanfrage wurde von beiden verarbeitet, und die Ergebnisse wurden verglichen und protokolliert. Diskrepanzen wurden untersucht, und die neue Engine wurde korrigiert. Erst nach sechs Wochen Übereinstimmung im Shadow-Modus über alle Falltypen hinweg schaltete das Team den Toggle um, um Live-Verkehr zur neuen Engine zu leiten. Die alte Engine blieb drei Monate lang hinter dem Toggle als Fallback verfügbar.

Ein Logistikunternehmen migrierte seinen Preisdienst von einer monolithischen Delphi-Anwendung zu einer modernen REST-API. Die Preislogik war komplex, variierte je nach Kundenvertrag und hatte fünfzehn Jahre an Sonderfällen angesammelt. Das Team deployte die neue Preis-API neben dem alten System und nutzte einen Berechtigungs-Toggle, um bestimmte Kundenkonten zum neuen Dienst zu leiten. Beginnend mit intern verwalteten Konten mit geringem Volumen erweiterten sie die freigeschaltete Menge über zwölf Wochen schrittweise. Mehrere Grenzfälle tauchten während des Rollouts für bestimmte Vertragstypen auf; weil jeder betroffene Kunde bereits durch die Zielregeln des Toggles identifiziert war, konnte das Team den neuen Pfad nur für diese Konten deaktivieren, während der Fehler behoben wurde, ohne den Rest der ausgerollten Population zu beeinträchtigen.

Eine europäische Bank ersetzte ihr Zinsberechnungsmodul, das nachts als Teil eines Batch-Prozesses lief. Die Batch-Umgebung hatte keine automatisierte Deployment-Pipeline — Releases durchliefen einen vierwöchigen Änderungsgenehmigungsprozess. Das Team führte einen einfachen datenbankgestützten Toggle ein, den das Berechnungsmodul beim Start las. Neuer Berechnungscode wurde während eines Standard-Wartungsfensters deployt. Der Toggle blieb auf den alten Codepfad gesetzt, bis das Geschäft die Ergebnisse des Parallellaufs abgenommen hatte, woraufhin das Betriebsteam eine einzelne Zeile in der Konfigurationstabelle aktualisierte und der nächste Batch-Lauf die neue Implementierung verwendete. Kein zusätzliches Deployment war nötig, und der Genehmigungsprozess war bereits Wochen zuvor erfüllt worden, als der Code im inaktiven Zustand deployt wurde.
