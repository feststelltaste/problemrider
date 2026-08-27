---
title: Produktionsnahe Testdaten
description: Aufbau von Testdaten aus anonymisierten Produktionsdaten,
  sodass Tests auf die unordentlichen, historisch gewachsenen Datensätze
  treffen, die synthetische Daten nie enthalten.
category:
- Testing
- Database
- Process
problems:
- inadequate-test-data-management
- testing-complexity
- inadequate-test-infrastructure
- testing-environment-fragility
- insufficient-testing
- flaky-tests
- high-defect-rate-in-production
- regression-bugs
- data-migration-complexities
- data-migration-integrity-issues
- poor-test-coverage
- increased-manual-testing-effort
- increased-bug-count
- schema-evolution-paralysis
- test-debt
- incorrect-index-type
layout: solution
lang: de
en_slug: production-like-test-data
related_solutions:
- slug: mass-test-data-generation
  similarity: 0.8
- slug: simulation-environments
  similarity: 0.65
- slug: parallel-run
  similarity: 0.65
- slug: test-coverage-strategy
  similarity: 0.65
- slug: datensparsamkeit
  similarity: 0.65
- slug: characterization-tests
  similarity: 0.6
---

## Description

Produktionsnahe Testdaten bedeutet, Testdatensätze aus echten Produktionsdaten zu konstruieren, anonymisiert und reduziert, statt sie synthetisch zu generieren. Der Grund ist, dass synthetische Daten aus dem Schema und aus dem Domänenmodell des Entwicklers generiert werden — was bedeutet, dass sie genau die Fälle enthalten, an die der Entwickler bereits gedacht hat, und keinen der Fälle, die Legacy-Systeme zum Scheitern bringen. Echte Daten in einem alten System sind weit merkwürdiger: Datensätze, die erstellt wurden, bevor eine Spalte existierte, Kodierungen aus einer Migration von 2009, Kunden mit Namen, die die Validierung heute ablehnen würde, und Zustände, die der aktuelle Code für unmöglich hält. Dies sind die Eingaben, die in Produktion Dinge zerbrechen und nie in einer Testsuite erscheinen. Anonymisierte Produktionsdaten bringen diese Verteilung ins Testen, was üblicherweise die schnellste verfügbare Verbesserung des Realismus einer Legacy-Testumgebung ist.

## How to Apply ◆

> Die wertvollste Eigenschaft von Produktionsdaten in einem Legacy-System ist ihre Geschichte: zwanzig Jahre Datensätze, geschrieben von einem Dutzend Versionen der Anwendung, von denen mehrere unterschiedliche Vorstellungen davon hatten, was gültig war.

- **Anonymisieren Sie, bevor die Daten die Produktion verlassen**, im Extraktionsschritt, nicht nachdem sie irgendwo anders gelandet sind. Jede Architektur, in der rohe Produktionsdaten „vorübergehend" eine niedrigere Umgebung erreichen, leckt letztlich, und dies ist sowohl eine rechtliche als auch eine Reputationsexposition.
- **Bewahren Sie die Form, während Sie die Werte ersetzen.** Referentielle Integrität, Kardinalitäten, Verteilungen und Grenzfallstruktur müssen die Anonymisierung überleben, sonst verliert der Datensatz genau das, was ihn wertvoll machte. Jeden Namen durch „Test User" zu ersetzen zerstört die Kodierungs- und Längenfälle, die der springende Punkt waren.
- Verwenden Sie **konsistente Pseudonymisierung**, sodass derselbe echte Wert überall demselben Ersatz zugeordnet wird. Ohne das brechen Joins, und mehrtabellige Szenarien werden untestbar.
- **Behandeln Sie indirekt identifizierende Daten**, nicht nur Namen und Identifikatoren. Eine seltene Kombination aus Postleitzahl, Geburtsdatum und Produkt kann eine Person ebenso effektiv identifizieren wie ein Name, besonders in kleinen Populationen, und naive Anonymisierung übersieht dies routinemäßig.
- **Reduzieren Sie das Volumen, während Sie die Vielfalt behalten.** Eine zufällige Ein-Prozent-Stichprobe verliert die seltenen Fälle, die am meisten zählen. Sampeln Sie, indem Sie einen Ausschnitt gewöhnlicher Datensätze plus eine bewusste Erfassung jedes distinkten Enumerationswerts, Grenzdatums und ungewöhnlichen im vollständigen Datensatz vorhandenen Zustands nehmen.
- **Automatisieren Sie die Auffrischung nach einem Zeitplan**, sodass die Testdaten verfolgen, wie sich Produktion weiterentwickelt. Ein einmal extrahierter und drei Jahre lang genutzter Datensatz hört langsam auf, dem System zu ähneln, und die Ähnlichkeit war die gesamte Rechtfertigung.
- **Beziehen Sie die Datenschutzfunktion früh ein** und dokumentieren Sie den Anonymisierungsansatz. Dies ist in den meisten Rechtsordnungen eine rechtliche Frage, und ein nie überprüfter Ansatz wird tendenziell während eines Audits entdeckt statt in einer Design-Diskussion.
- **Kombinieren Sie mit synthetischer Generierung**, statt sie zu ersetzen. Anonymisierte Daten decken das Historische und Merkwürdige ab; synthetische Generierung deckt Volumen für Lasttests und neue Fälle ab, die Produktion noch nicht produziert hat.
- **Behandeln Sie die Extraktionspipeline als Produktionscode** — überprüft, getestet und versionskontrolliert. Ein Defekt in der Anonymisierung ist eine Datenschutzverletzung, kein Testfehlschlag.

## Tradeoffs ⇄

> Echte Daten finden die Defekte, die synthetische Daten strukturell nicht finden können, auf Kosten eines echten Datenschutzrisikos und einer Pipeline, die gebaut und gepflegt werden muss.

**Vorteile:**

- Tests treffen auf die historischen und fehlgeformten Datensätze, die Legacy-Systeme tatsächlich zerbrechen und an die kein Entwickler denken würde, sie zu generieren.
- Datenmigrationen können gegen realistische Eingaben geprobt werden, was der Ursprung der meisten Migrationsfehlschläge ist — die Datensätze, die Annahmen verletzen, von denen niemand wusste, dass er sie traf.
- Abfrageperformance verhält sich realistisch, da Datenverteilung und -volumen Ausführungspläne auf Weisen antreiben, die uniforme synthetische Daten nicht reproduzieren.
- In Produktion gefundene Defekte können in einer Testumgebung reproduziert werden, was oft unmöglich ist, wenn die Testdaten keinen vergleichbaren Fall haben.
- Undokumentierte Datenzustände werden sichtbar, und jeder gefundene ist ein Stück der tatsächlichen Spezifikation des Systems, das wiedergewonnen wurde.

**Kosten und Risiken:**

- Anonymisierung kann fehlschlagen. Unvollständige Anonymisierung in einer niedrigeren Umgebung ist eine Datenschutzverletzung, und niedrigere Umgebungen haben schwächere Zugangskontrollen, gerade weil angenommen wurde, dass sie keine echten Daten halten.
- Indirekte Identifikation ist subtil und leicht falsch zu machen, besonders bei kleinen Populationen oder seltenen Attributkombinationen.
- Die Extraktions- und Anonymisierungspipeline ist echte Software, die gepflegt werden muss, während sich das Schema weiterentwickelt, und sie bricht still, wenn eine neue Spalte erscheint.
- Realistische Volumina machen Testumgebungen größer und langsamer, was gegen das schnelle Feedback drückt, das Tests brauchen.
- Manche Rechtsordnungen und Sektoren beschränken diesen Ansatz stark, unabhängig von der Anonymisierungsqualität, und die rechtliche Prüfung kann länger dauern als der Bau der Pipeline.

## How It Could Be

Ein Team, das ein Rentenverwaltungssystem betreute, hatte einen synthetischen Testdatensatz von 500 Mitgliedern, alle mit wohlgeformten Datensätzen, generiert aus dem aktuellen Schema. Produktion hielt 340.000 Mitglieder mit Datensätzen, die bis 1987 zurückreichten. Ihr Fehlermuster war konsistent: Änderungen bestanden alle Tests und scheiterten dann in Produktion an Datensätzen, die vor einer Schemaänderung datierten. Sie bauten eine Anonymisierungspipeline, die einen 4.000-Mitglieder-Extrakt produzierte, der bewusst jede distinkte Kombination aus Schematyp, Status und Beitragsgeschichte einschloss, die in der vollständigen Population vorhanden war. Der erste Lauf der bestehenden Testsuite gegen den neuen Datensatz produzierte 31 Fehlschläge, alle echt: unbehandelte Null-Beitragsperioden, zwei Datumsformate, von denen der Code annahm, sie seien wegmigriert worden, und eine Mitgliederkategorie, die 1998 für neue Teilnehmer geschlossen worden war und die drei Codepfade nicht behandelten. Produktionsdefekte, die unerwarteten Daten zugeschrieben wurden, fielen über die folgenden zwei Quartale um etwa siebzig Prozent.

Derselbe Datensatz änderte, wie das Team eine nachfolgende Schemamigration anging. Ihre vorherige Migration war gegen die synthetischen Daten getestet worden, lief sauber und scheiterte dann in Produktion an 1.200 Datensätzen, was um vier Uhr morgens einen Rollback erforderte. Die Probe der nächsten Migration gegen den anonymisierten Extrakt fand vier Fehlerklassen, bevor die Änderung überhaupt geplant war, einschließlich einer Reihe von Datensätzen, deren Fremdschlüssel auf eine Tabellenzeile zeigte, die Jahre zuvor gelöscht worden war — ein Zustand, den das Schema verbot und den die Daten dennoch enthielten. Die Migration lief in Produktion ohne Vorfall, und das Problem der verwaisten Datensätze wurde als eigenes Arbeitsstück behoben statt als Notfall.
